from collections.abc import Callable
from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct

from jax_baselines.DDPG.base_class import (
    Deteministic_Policy_Gradient_Family,
    merge_actor_metrics,
)
from jax_baselines.DDPG.metrics import critic_metrics
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.param_updates import scaled_by_reset, soft_update
from jax_baselines.optim import optimizer_metrics


@struct.dataclass
class TD3CheckpointParams:
    policy_params: Any
    critic_params: Any
    target_policy_params: Any
    target_critic_params: Any


class TD3(Deteministic_Policy_Gradient_Family):
    _run_name = "TD3"

    @property
    def _actor_schedule(self):
        return self.policy_delay, 0

    def __init__(
        self,
        env_builder: Callable,
        model_builder_maker,
        target_action_noise_mul=2.0,
        action_noise=0.1,
        policy_delay=2,
        **kwargs,
    ):

        self.action_noise = action_noise
        self.target_action_noise = action_noise * target_action_noise_mul
        self.action_noise_clamp = 0.5
        self.policy_delay = policy_delay

        super().__init__(env_builder, model_builder_maker, **kwargs)

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.policy_kwargs,
        )
        (
            self.actor,
            self.critic,
            self.policy_params,
            self.critic_params,
        ) = model_builder(next(self.key_seq), print_model=True)
        self.target_policy_params = deepcopy(self.policy_params)
        self.target_critic_params = deepcopy(self.critic_params)

        self.opt_policy_state = self.optimizer.init(self.policy_params)
        self.opt_critic_state = self.optimizer.init(self.critic_params)

    def checkpoint_params(self):
        return TD3CheckpointParams(
            policy_params=self.policy_params,
            critic_params=self.critic_params,
            target_policy_params=self.target_policy_params,
            target_critic_params=self.target_critic_params,
        )

    def load_checkpoint_params(self, bundle):
        self.policy_params = bundle.policy_params
        self.critic_params = bundle.critic_params
        self.target_policy_params = bundle.target_policy_params
        self.target_critic_params = bundle.target_critic_params

    def _get_actions(self, state, obses, key):
        key, noise_key = jax.random.split(key)
        actions = self._get_eval_actions(state, obses)
        noise = self.action_noise * jax.random.normal(noise_key, actions.shape)
        return jnp.clip(actions + noise, -1, 1), key

    def _get_eval_actions(self, state, obses):
        return self.actor(state["policy"], None, convert_normalized_obs(obses))

    @property
    def _train_state(self):
        return (
            self.policy_params,
            self.critic_params,
            self.target_policy_params,
            self.target_critic_params,
            self.opt_policy_state,
            self.opt_critic_state,
        )

    @_train_state.setter
    def _train_state(self, state):
        (
            self.policy_params,
            self.critic_params,
            self.target_policy_params,
            self.target_critic_params,
            self.opt_policy_state,
            self.opt_critic_state,
        ) = state

    def _train_step(
        self, state, key, step, flags, obses, actions, rewards, nxtobses, terminateds, weights=1
    ):
        (
            policy_params,
            critic_params,
            target_policy_params,
            target_critic_params,
            opt_policy_state,
            opt_critic_state,
        ) = state
        obses = convert_normalized_obs(obses)
        nxtobses = convert_normalized_obs(nxtobses)
        not_terminateds = 1.0 - terminateds
        targets = self._target(
            target_policy_params,
            target_critic_params,
            rewards,
            nxtobses,
            not_terminateds,
            key,
        )
        (critic_loss, (abs_error, metrics)), grad = jax.value_and_grad(
            self._critic_loss, has_aux=True
        )(critic_params, policy_params, obses, actions, targets, weights, key, flags.diagnostics)
        updates, opt_critic_state = self.optimizer.update(
            grad, opt_critic_state, params=critic_params, diagnostics=flags.diagnostics
        )
        critic_params = optax.apply_updates(critic_params, updates)

        actor_metrics = {}
        if flags.actor:
            actor_loss, grad = jax.value_and_grad(self._actor_loss)(
                policy_params, critic_params, obses, key
            )
            updates, opt_policy_state = self.optimizer.update(
                grad, opt_policy_state, params=policy_params, diagnostics=flags.diagnostics
            )
            policy_params = optax.apply_updates(policy_params, updates)
            target_policy_params = soft_update(
                policy_params, target_policy_params, self.target_network_update_tau
            )
            target_critic_params = soft_update(
                critic_params, target_critic_params, self.target_network_update_tau
            )
            if flags.diagnostics:
                actor_metrics = {
                    "loss/actor_loss": actor_loss,
                    "loss/actor_q_mean": -actor_loss,
                    **optimizer_metrics(opt_policy_state, "actor"),
                }
        elif flags.diagnostics:
            actor_metrics = dict.fromkeys(
                (
                    "loss/actor_loss",
                    "loss/actor_q_mean",
                    *optimizer_metrics(opt_policy_state, "actor"),
                ),
                jnp.asarray(0.0),
            )
        if flags.diagnostics:
            metrics.update(optimizer_metrics(opt_critic_state, "critic"))
            metrics["loss/targets"] = jnp.mean(targets)
        metrics, metric_counts = merge_actor_metrics(
            {**metrics, "loss/qloss": critic_loss}, actor_metrics, flags.actor
        )

        policy_params, opt_policy_state = scaled_by_reset(
            policy_params,
            opt_policy_state,
            self.optimizer,
            key,
            flags.reset,
            0.1,  # tau = 0.1 is softreset, but original paper uses 1.0
        )
        critic_params, opt_critic_state = scaled_by_reset(
            critic_params,
            opt_critic_state,
            self.optimizer,
            key,
            flags.reset,
            0.1,  # tau = 0.1 is softreset, but original paper uses 1.0
        )
        return (
            policy_params,
            critic_params,
            target_policy_params,
            target_critic_params,
            opt_policy_state,
            opt_critic_state,
        ), (abs_error if self.prioritized_replay else None, metrics, metric_counts)

    def _critic_loss(
        self, critic_params, policy_params, obses, actions, targets, weights, key, diagnostics
    ):
        q1, q2 = self.critic(critic_params, policy_params, key, obses, actions)
        error1 = jnp.squeeze(q1 - targets)
        error2 = jnp.squeeze(q2 - targets)
        critic_loss = jnp.mean(weights * jnp.square(error1)) + jnp.mean(
            weights * jnp.square(error2)
        )
        metrics = (
            critic_metrics(
                (q1, q2),
                targets,
                (jnp.square(error1), jnp.square(error2)),
                weights,
                jnp.abs(error1) if self.prioritized_replay else None,
            )
            if diagnostics
            else {}
        )
        return critic_loss, (jnp.abs(error1), metrics)

    def _actor_loss(self, policy_params, critic_params, obses, key):
        actions = self.actor(policy_params, key, obses)
        q1, _ = self.critic(critic_params, policy_params, key, obses, actions)
        return -jnp.mean(q1)

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
                * jax.random.normal(key, (self.batch_size, self.action_size[0])),
                -self.action_noise_clamp,
                self.action_noise_clamp,
            ),
            -1.0,
            1.0,
        )
        q1, q2 = self.critic(target_critic_params, target_policy_params, key, nxtobses, next_action)
        next_q = jnp.minimum(q1, q2)
        return (not_terminateds * next_q * self._gamma) + rewards
