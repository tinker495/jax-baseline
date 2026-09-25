from copy import deepcopy

import jax
import jax.numpy as jnp
import optax

from jax_baselines.APE_X.dpg_base_class import Ape_X_Deteministic_Policy_Gradient_Family
from jax_baselines.core.bulk_training import SCAN_UNROLL
from jax_baselines.DDPG.ou_noise import ou_step
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.param_updates import soft_update


class APE_X_DDPG(Ape_X_Deteministic_Policy_Gradient_Family):
    _run_name = "Ape_X_DDPG"

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
                nxtobses = convert_normalized_obs(nxtobses)
                next_action = actor(params["policy"], key, nxtobses)
                next_q = critic(params["critic"], params["policy"], key, nxtobses, next_action)
                q_values = critic(
                    params["critic"], params["policy"], key, convert_normalized_obs(obses), actions
                )
                target = rewards + gamma * (1.0 - terminateds) * next_q
                td_error = q_values - target
                return jnp.squeeze(jnp.abs(td_error))

            def noise_step(noise, key, reset):
                # OU exploration noise; a new episode restarts it from a fresh N(0, 0.2) draw.
                if reset:
                    key, reset_key = jax.random.split(key)
                    noise = 0.2 * jax.random.normal(reset_key, noise.shape)
                return ou_step(noise, key)

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
        del step  # DDPG has no step-dependent schedule.
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
            policy_params, critic_params, opt_policy_state, opt_critic_state, key = carry
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
            )
            actor_updates, opt_policy_state = self.optimizer.update(
                actor_grad, opt_policy_state, params=policy_params, diagnostics=False
            )
            critic_updates, opt_critic_state = self.optimizer.update(
                critic_grad, opt_critic_state, params=critic_params, diagnostics=False
            )
            return (
                optax.apply_updates(policy_params, actor_updates),
                optax.apply_updates(critic_params, critic_updates),
                opt_policy_state,
                opt_critic_state,
                key,
            ), (critic_loss, actor_loss, abs_error)

        (
            (
                policy_params,
                critic_params,
                opt_policy_state,
                opt_critic_state,
                key,
            ),
            (critic_loss, actor_loss, abs_error),
        ) = jax.lax.scan(
            f,
            (policy_params, critic_params, opt_policy_state, opt_critic_state, key),
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

    def _loss(self, policy_params, critic_params, obses, actions, targets, weights, key):
        vals = self.critic(critic_params, policy_params, key, obses, actions)
        error = jnp.squeeze(vals - targets)
        critic_loss = jnp.mean(jnp.square(error) * weights)
        policy = self.actor(policy_params, key, obses)
        vals = self.critic(jax.lax.stop_gradient(critic_params), policy_params, key, obses, policy)
        actor_loss = jnp.mean(-vals)
        total_loss = critic_loss + actor_loss
        return total_loss, (critic_loss, -actor_loss, jnp.abs(error))

    def _target(
        self,
        target_policy_params,
        target_critic_params,
        rewards,
        nxtobses,
        not_terminateds,
        key,
    ):
        next_action = self.actor(target_policy_params, key, nxtobses)
        next_q = self.critic(target_critic_params, target_policy_params, key, nxtobses, next_action)
        return (not_terminateds * next_q * self._gamma) + rewards
