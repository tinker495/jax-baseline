from copy import deepcopy

import jax
import jax.numpy as jnp
import optax

from jax_baselines.APE_X.dpg_base_class import Ape_X_Deteministic_Policy_Gradient_Family
from jax_baselines.core.bulk_training import SCAN_UNROLL
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.param_updates import soft_update


class APE_X_TD3(Ape_X_Deteministic_Policy_Gradient_Family):
    _run_name = "Ape_X_TD3"

    def __init__(
        self,
        workers,
        model_builder_maker,
        runtime,
        target_action_noise_mul=1.5,
        policy_delay=3,
        gamma=0.995,
        learning_rate=5e-5,
        buffer_size=50000,
        exploration_initial_eps=0.9,
        exploration_decay=0.7,
        batch_num=16,
        mini_batch_size=512,
        n_step=1,
        learning_starts=1000,
        target_network_update_tau=5e-4,
        gradient_steps=1,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-3,
        scaled_by_reset=False,
        log_interval=200,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        seed=None,
        optimizer_factory=None,
        compress_memory=False,
        param_broadcast_freq=20,
        apex_replay_factory=None,
        checkpoint_store=None,
    ):
        self.action_noise = exploration_initial_eps ** (1 + exploration_decay)
        self.target_action_noise = self.action_noise * target_action_noise_mul
        self.action_noise_clamp = 0.5
        self.policy_delay = policy_delay

        super().__init__(
            workers,
            model_builder_maker,
            runtime=runtime,
            gamma=gamma,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            exploration_initial_eps=exploration_initial_eps,
            exploration_decay=exploration_decay,
            batch_num=batch_num,
            mini_batch_size=mini_batch_size,
            n_step=n_step,
            learning_starts=learning_starts,
            target_network_update_tau=target_network_update_tau,
            gradient_steps=gradient_steps,
            prioritized_replay_alpha=prioritized_replay_alpha,
            prioritized_replay_beta0=prioritized_replay_beta0,
            prioritized_replay_eps=prioritized_replay_eps,
            scaled_by_reset=scaled_by_reset,
            log_interval=log_interval,
            log_dir=log_dir,
            _init_setup_model=_init_setup_model,
            policy_kwargs=policy_kwargs,
            seed=seed,
            optimizer_factory=optimizer_factory,
            compress_memory=compress_memory,
            param_broadcast_freq=param_broadcast_freq,
            apex_replay_factory=apex_replay_factory,
            checkpoint_store=checkpoint_store,
        )

    def setup_model(self):
        self.model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.policy_kwargs,
        )
        self.actor_builder = self.get_actor_builder()

        self.actor, self.critic, self.policy_params, self.critic_params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.target_policy_params = deepcopy(self.policy_params)
        self.target_critic_params = deepcopy(self.critic_params)

        self.opt_policy_state = self.optimizer.init(self.policy_params)
        self.opt_critic_state = self.optimizer.init(self.critic_params)

    def get_actor_builder(self):
        gamma = self._gamma
        action_size = self.action_size[0]
        action_noise_clamp = self.action_noise_clamp
        target_action_noise = self.target_action_noise

        def builder():
            def get_abs_td_error(
                actor,
                critic,
                params,
                obses,
                actions,
                rewards,
                nxtobses,
                terminateds,
                key,
            ):
                size = next(iter(obses.values())).shape[0]
                nxtobses = convert_normalized_obs(nxtobses)
                next_action = jnp.clip(
                    actor(params["policy"], key, nxtobses)
                    + jnp.clip(
                        target_action_noise * jax.random.normal(key, (size, action_size)),
                        -action_noise_clamp,
                        action_noise_clamp,
                    ),
                    -1.0,
                    1.0,
                )
                q1, q2 = critic(params["critic"], params["policy"], key, nxtobses, next_action)
                next_q = jnp.minimum(q1, q2)
                q_values1, _ = critic(
                    params["critic"], params["policy"], key, convert_normalized_obs(obses), actions
                )
                target = rewards + gamma * (1.0 - terminateds) * next_q
                td1_error = jnp.abs(q_values1 - target)
                return jnp.squeeze(td1_error)

            def noise_step(noise, key, reset):
                # Stateless Gaussian exploration noise: episode boundaries do not matter.
                return jax.random.normal(key, noise.shape)

            return get_abs_td_error, noise_step

        return builder

    def _train_step(
        self,
        policy_params,
        critic_params,
        target_policy_params,
        target_critic_params,
        opt_policy_state,
        opt_critic_state,
        step,
        key,
        obses,
        actions,
        rewards,
        nxtobses,
        terminateds,
        weights=1,
        indexes=None,
    ):
        obses = convert_normalized_obs(obses)
        nxtobses = convert_normalized_obs(nxtobses)
        not_terminateds = 1.0 - terminateds
        batch_idxes = jnp.arange(self.batch_size).reshape(-1, self.mini_batch_size)
        obses_batch = jax.tree.map(lambda value: value[batch_idxes], obses)
        actions_batch = actions[batch_idxes]
        rewards_batch = rewards[batch_idxes]
        nxtobses_batch = jax.tree.map(lambda value: value[batch_idxes], nxtobses)
        not_terminateds_batch = not_terminateds[batch_idxes]
        weights_batch = jnp.broadcast_to(jnp.asarray(weights), (self.batch_size,))[batch_idxes]

        def f(carry, data):
            (
                policy_params,
                critic_params,
                opt_policy_state,
                opt_critic_state,
                key,
                step,
            ) = carry
            obses, actions, rewards, nxtobses, not_terminateds, weights = data
            key, *subkeys = jax.random.split(key, 3)
            targets = self._target(
                target_policy_params,
                target_critic_params,
                rewards,
                nxtobses,
                not_terminateds,
                subkeys[0],
            )
            (_, (critic_loss, actor_loss, abs_error)), (
                actor_grad,
                critic_grad,
            ) = jax.value_and_grad(self._loss, argnums=(0, 1), has_aux=True)(
                policy_params,
                critic_params,
                obses,
                actions,
                targets,
                weights,
                subkeys[1],
                step,
            )
            critic_updates, opt_critic_state = self.optimizer.update(
                critic_grad, opt_critic_state, params=critic_params, diagnostics=False
            )

            actor_updates, updated_policy_state = self.optimizer.update(
                actor_grad, opt_policy_state, params=policy_params, diagnostics=False
            )
            # A select, not lax.cond: a GPU conditional copies its predicate to the host on
            # every mini-update, while the actor gradient is computed either way.
            policy_params, opt_policy_state = jax.tree.map(
                lambda updated, kept: jnp.where(step % self.policy_delay == 0, updated, kept),
                (optax.apply_updates(policy_params, actor_updates), updated_policy_state),
                (policy_params, opt_policy_state),
            )
            return (
                policy_params,
                optax.apply_updates(critic_params, critic_updates),
                opt_policy_state,
                opt_critic_state,
                key,
                step + 1,
            ), (critic_loss, actor_loss, abs_error)

        (
            (
                policy_params,
                critic_params,
                opt_policy_state,
                opt_critic_state,
                key,
                step,
            ),
            (
                critic_loss,
                actor_loss,
                abs_error,
            ),
        ) = jax.lax.scan(
            f,
            (
                policy_params,
                critic_params,
                opt_policy_state,
                opt_critic_state,
                key,
                step,
            ),
            (
                obses_batch,
                actions_batch,
                rewards_batch,
                nxtobses_batch,
                not_terminateds_batch,
                weights_batch,
            ),
            unroll=SCAN_UNROLL,
        )
        target_policy_params = soft_update(
            policy_params, target_policy_params, self.target_network_update_tau
        )
        target_critic_params = soft_update(
            critic_params, target_critic_params, self.target_network_update_tau
        )
        new_priorities = jnp.reshape(abs_error, (-1,))
        return (
            policy_params,
            critic_params,
            target_policy_params,
            target_critic_params,
            opt_policy_state,
            opt_critic_state,
            jnp.mean(critic_loss),
            -jnp.mean(actor_loss),
            new_priorities,
        )

    def _loss(self, policy_params, critic_params, obses, actions, targets, weights, key, step):
        q1, q2 = self.critic(critic_params, policy_params, key, obses, actions)
        error1 = jnp.squeeze(q1 - targets)
        error2 = jnp.squeeze(q2 - targets)
        critic_loss = jnp.mean(weights * jnp.square(error1)) + jnp.mean(
            weights * jnp.square(error2)
        )
        policy = self.actor(policy_params, key, obses)
        vals, _ = self.critic(
            jax.lax.stop_gradient(critic_params), policy_params, key, obses, policy
        )
        actor_loss = jnp.mean(-vals)
        total_loss = jax.lax.select(
            step % self.policy_delay == 0, critic_loss + actor_loss, critic_loss
        )
        return total_loss, (critic_loss, actor_loss, jnp.abs(error1))

    def _target(
        self,
        target_policy_params,
        target_critic_params,
        rewards,
        nxtobses,
        not_terminateds,
        key,
    ):
        next_action = jnp.clip(
            self.actor(target_policy_params, key, nxtobses)
            + jnp.clip(
                self.target_action_noise
                * jax.random.normal(key, (self.mini_batch_size, self.action_size[0])),
                -self.action_noise_clamp,
                self.action_noise_clamp,
            ),
            -1.0,
            1.0,
        )
        q1, q2 = self.critic(target_critic_params, target_policy_params, key, nxtobses, next_action)
        next_q = jnp.minimum(q1, q2)
        return (not_terminateds * next_q * self._gamma) + rewards
