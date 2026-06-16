from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.APE_X.dpg_base_class import Ape_X_Deteministic_Policy_Gradient_Family
from jax_baselines.core.seeding import key_gen
from jax_baselines.math.jax_utils import convert_jax
from jax_baselines.math.param_updates import soft_update


class APE_X_TD3(Ape_X_Deteministic_Policy_Gradient_Family):
    _run_name = "Ape_X_TD3"

    def __init__(
        self,
        workers,
        model_builder_maker,
        manager=None,
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
        simba=False,
        log_interval=200,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        seed=None,
        optimizer_factory=None,
        compress_memory=False,
        param_broadcast_freq=20,
        multi_replay_factory=None,
        worker_replay_factory=None,
    ):
        self.action_noise = exploration_initial_eps ** (1 + exploration_decay)
        self.target_action_noise = self.action_noise * target_action_noise_mul
        self.action_noise_clamp = 0.5
        self.policy_delay = policy_delay

        super().__init__(
            workers,
            model_builder_maker,
            manager=manager,
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
            simba=simba,
            log_interval=log_interval,
            log_dir=log_dir,
            _init_setup_model=_init_setup_model,
            policy_kwargs=policy_kwargs,
            seed=seed,
            optimizer_factory=optimizer_factory,
            compress_memory=compress_memory,
            param_broadcast_freq=param_broadcast_freq,
            multi_replay_factory=multi_replay_factory,
            worker_replay_factory=worker_replay_factory,
        )

    def setup_model(self):
        self.model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.policy_kwargs,
        )
        self.actor_builder = self.get_actor_builder()

        self.preproc, self.actor, self.critic, self.params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)

        self._get_actions = jax.jit(self._get_actions)
        self._train_step = jax.jit(self._train_step)

    def get_actor_builder(self):
        gamma = self._gamma
        action_size = self.action_size[0]
        action_noise_clamp = self.action_noise_clamp
        target_action_noise = self.target_action_noise

        class Noise:
            def __init__(self, action_size) -> None:
                self.action_size = action_size

            def __call__(self):
                return np.random.normal(size=(1, self.action_size))

            def reset(self, worker_id):
                pass

        def builder():
            noise = Noise(action_size)
            key_seq = key_gen(42)

            def get_abs_td_error(
                actor,
                critic,
                preproc,
                params,
                obses,
                actions,
                rewards,
                nxtobses,
                terminateds,
                key,
            ):
                size = obses[0].shape[0]
                next_feature = preproc(params, key, convert_jax(nxtobses))
                next_action = jnp.clip(
                    actor(params, key, next_feature)
                    + jnp.clip(
                        target_action_noise * jax.random.normal(key, (size, action_size)),
                        -action_noise_clamp,
                        action_noise_clamp,
                    ),
                    -1.0,
                    1.0,
                )
                q1, q2 = critic(params, key, next_feature, next_action)
                next_q = jnp.minimum(q1, q2)
                feature = preproc(params, key, convert_jax(obses))
                q_values1, _ = critic(params, key, feature, actions)
                target = rewards + gamma * (1.0 - terminateds) * next_q
                td1_error = jnp.abs(q_values1 - target)
                return jnp.squeeze(td1_error)

            def actor(actor, preproc, params, obses, key):
                return actor(params, key, preproc(params, key, convert_jax(obses)))

            def get_action(actor, params, obs, noise, epsilon, key):
                actions = np.clip(np.asarray(actor(params, obs, key)) + noise() * epsilon, -1, 1)[0]
                return actions

            def random_action(params, obs, noise, epsilon, key):
                return np.random.uniform(-1.0, 1.0, size=(action_size))

            return get_abs_td_error, actor, get_action, random_action, noise, key_seq

        return builder

    def _invoke_train_step(self, steps, data):
        return self._train_step(
            self.params, self.target_params, self.opt_state, next(self.key_seq), steps, **data
        )

    def _train_step(
        self,
        params,
        target_params,
        opt_state,
        key,
        step,
        obses,
        actions,
        rewards,
        nxtobses,
        terminateds,
        weights=1,
        indexes=None,
    ):
        obses = convert_jax(obses)
        nxtobses = convert_jax(nxtobses)
        not_terminateds = 1.0 - terminateds
        batch_idxes = jnp.arange(self.batch_size).reshape(-1, self.mini_batch_size)
        obses_batch = [o[batch_idxes] for o in obses]
        actions_batch = actions[batch_idxes]
        rewards_batch = rewards[batch_idxes]
        nxtobses_batch = [o[batch_idxes] for o in nxtobses]
        not_terminateds_batch = not_terminateds[batch_idxes]
        weights_batch = weights[batch_idxes]

        def f(carry, data):
            params, opt_state, key, step = carry
            obses, actions, rewards, nxtobses, not_terminateds, weights = data
            key, *subkeys = jax.random.split(key, 3)
            targets = self._target(target_params, rewards, nxtobses, not_terminateds, subkeys[0])
            (total_loss, (critic_loss, actor_loss, abs_error)), grad = jax.value_and_grad(
                self._loss, has_aux=True
            )(params, obses, actions, targets, weights, subkeys[1], step)
            updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            step = step + 1
            return (params, opt_state, key, step), (total_loss, critic_loss, actor_loss, abs_error)

        (params, opt_state, key, step), (
            total_loss,
            critic_loss,
            actor_loss,
            abs_error,
        ) = jax.lax.scan(
            f,
            (params, opt_state, key, step),
            (
                obses_batch,
                actions_batch,
                rewards_batch,
                nxtobses_batch,
                not_terminateds_batch,
                weights_batch,
            ),
        )
        target_params = soft_update(params, target_params, self.target_network_update_tau)
        new_priorities = jnp.reshape(abs_error, (-1,))
        return (
            params,
            target_params,
            opt_state,
            jnp.mean(critic_loss),
            -jnp.mean(actor_loss),
            new_priorities,
        )

    def _loss(self, params, obses, actions, targets, weights, key, step):
        feature = self.preproc(params, key, obses)
        q1, q2 = self.critic(params, key, feature, actions)
        error1 = jnp.squeeze(q1 - targets)
        error2 = jnp.squeeze(q2 - targets)
        critic_loss = jnp.mean(weights * jnp.square(error1)) + jnp.mean(
            weights * jnp.square(error2)
        )
        policy = self.actor(params, key, feature)
        vals, _ = self.critic(jax.lax.stop_gradient(params), key, feature, policy)
        actor_loss = jnp.mean(-vals)
        total_loss = jax.lax.select(
            step % self.policy_delay == 0, critic_loss + actor_loss, critic_loss
        )
        return total_loss, (critic_loss, actor_loss, jnp.abs(error1))

    def _target(self, target_params, rewards, nxtobses, not_terminateds, key):
        next_feature = self.preproc(target_params, key, nxtobses)
        next_action = jnp.clip(
            self.actor(target_params, key, next_feature)
            + jnp.clip(
                self.target_action_noise
                * jax.random.normal(key, (self.mini_batch_size, self.action_size[0])),
                -self.action_noise_clamp,
                self.action_noise_clamp,
            ),
            -1.0,
            1.0,
        )
        q1, q2 = self.critic(target_params, key, next_feature, next_action)
        next_q = jnp.minimum(q1, q2)
        return (not_terminateds * next_q * self._gamma) + rewards
