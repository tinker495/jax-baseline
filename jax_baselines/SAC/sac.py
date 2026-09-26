from collections.abc import Callable
from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from jax_baselines.DDPG.base_class import (
    Deteministic_Policy_Gradient_Family,
    merge_actor_metrics,
)
from jax_baselines.DDPG.metrics import critic_metrics, stochastic_actor_metrics
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.param_updates import scaled_by_reset, soft_update
from jax_baselines.math.policy_math import entropy_target_from_sigma
from jax_baselines.optim import optimizer_metrics


def sample_action(mu, log_std, key):
    eps = jax.random.normal(key, mu.shape)
    return jnp.tanh(mu + jnp.exp(log_std) * eps)


def mode_action(mu):
    return jnp.tanh(mu)


@struct.dataclass
class SACCheckpointParams:
    policy_params: Any
    critic_params: Any
    target_critic_params: Any
    log_ent_coef: Any


class SAC(Deteministic_Policy_Gradient_Family):
    _run_name = "SAC"

    @property
    def _actor_schedule(self):
        return self.actor_update_period, 1

    def __init__(
        self,
        env_builder: Callable,
        model_builder_maker,
        ent_coef="auto_0.01",
        sigma_target=0.15,
        actor_update_period=2,
        **kwargs,
    ):
        if actor_update_period <= 0:
            raise ValueError("actor_update_period must be greater than 0")

        self._ent_coef = ent_coef
        self.ent_coef_learning_rate = 3e-4
        self.actor_update_period = actor_update_period

        super().__init__(env_builder, model_builder_maker, **kwargs)

        self.target_entropy = entropy_target_from_sigma(
            int(np.prod(self.action_size)), sigma_target
        )

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
        self.target_critic_params = deepcopy(self.critic_params)
        self.opt_policy_state = self.optimizer.init(self.policy_params)
        self.opt_critic_state = self.optimizer.init(self.critic_params)

        self._setup_entropy_coef()

    def checkpoint_params(self):
        return SACCheckpointParams(
            policy_params=self.policy_params,
            critic_params=self.critic_params,
            target_critic_params=self.target_critic_params,
            log_ent_coef=self.log_ent_coef,
        )

    def load_checkpoint_params(self, bundle):
        self.policy_params = bundle.policy_params
        self.critic_params = bundle.critic_params
        self.target_critic_params = bundle.target_critic_params
        self.log_ent_coef = bundle.log_ent_coef

    def _get_pi_log_prob(self, params, obses, key):
        mu, log_std = self.actor(params, None, obses)
        std = jnp.exp(log_std)
        x_t = mu + std * jax.random.normal(key, std.shape)
        pi = jax.nn.tanh(x_t)
        log_prob = jnp.sum(
            -0.5 * (jnp.square((x_t - mu) / (std + 1e-6)) + 2 * log_std + jnp.log(2 * np.pi))
            - jnp.log(1 - jnp.square(pi) + 1e-6),
            axis=1,
            keepdims=True,
        )
        return pi, log_prob, log_std

    def _get_actions(self, state, obses, key):
        key, sample_key = jax.random.split(key)
        mu, log_std = self.actor(state["policy"], None, convert_normalized_obs(obses))
        return sample_action(mu, log_std, sample_key), key

    def _get_eval_actions(self, state, obses):
        mu, _ = self.actor(state["policy"], None, convert_normalized_obs(obses))
        return mode_action(mu)

    @property
    def _train_state(self):
        return (
            self.policy_params,
            self.critic_params,
            self.target_critic_params,
            self.opt_policy_state,
            self.opt_critic_state,
            self.opt_ent_coef_state,
            self.log_ent_coef,
        )

    @_train_state.setter
    def _train_state(self, state):
        (
            self.policy_params,
            self.critic_params,
            self.target_critic_params,
            self.opt_policy_state,
            self.opt_critic_state,
            self.opt_ent_coef_state,
            self.log_ent_coef,
        ) = state

    def _train_step(
        self, state, key, step, flags, obses, actions, rewards, nxtobses, terminateds, weights=1
    ):
        (
            policy_params,
            critic_params,
            target_critic_params,
            opt_policy_state,
            opt_critic_state,
            opt_ent_coef_state,
            log_ent_coef,
        ) = state
        obses = convert_normalized_obs(obses)
        nxtobses = convert_normalized_obs(nxtobses)
        not_terminateds = 1.0 - terminateds
        ent_coef = jnp.exp(log_ent_coef)
        key1, key2, key3 = jax.random.split(key, 3)
        targets = self._target(
            policy_params,
            target_critic_params,
            rewards,
            nxtobses,
            not_terminateds,
            key1,
            ent_coef,
        )

        (critic_loss, (abs_error, metrics)), grad = jax.value_and_grad(
            self._critic_loss, has_aux=True
        )(critic_params, policy_params, obses, actions, targets, weights, key2, flags.diagnostics)
        updates, opt_critic_state = self.optimizer.update(
            grad, opt_critic_state, params=critic_params, diagnostics=flags.diagnostics
        )
        critic_params = optax.apply_updates(critic_params, updates)

        actor_metrics = {}
        if flags.actor:
            (actor_loss, (log_prob, actor_metrics)), grad = jax.value_and_grad(
                self._actor_loss, has_aux=True
            )(policy_params, critic_params, obses, key3, ent_coef, flags.diagnostics)
            updates, opt_policy_state = self.optimizer.update(
                grad, opt_policy_state, params=policy_params, diagnostics=flags.diagnostics
            )
            policy_params = optax.apply_updates(policy_params, updates)
            if flags.diagnostics:
                actor_metrics.update(
                    {"loss/actor_loss": actor_loss, **optimizer_metrics(opt_policy_state, "actor")}
                )
            if self.auto_entropy:
                log_ent_coef, opt_ent_coef_state, entropy_metrics = self._train_ent_coef(
                    log_ent_coef, opt_ent_coef_state, log_prob, flags.diagnostics
                )
                actor_metrics.update(entropy_metrics)
        elif flags.diagnostics:
            actor_metrics = self._skipped_actor_metrics(
                ent_coef, opt_policy_state, opt_ent_coef_state
            )
        metrics["loss/qloss"] = critic_loss
        if flags.diagnostics:
            metrics.update(optimizer_metrics(opt_critic_state, "critic"))
            metrics["loss/ent_coef"] = jnp.exp(log_ent_coef)
            metrics["loss/targets"] = jnp.mean(targets)
        metrics, metric_counts = merge_actor_metrics(metrics, actor_metrics, flags.actor)

        target_critic_params = soft_update(
            critic_params, target_critic_params, self.target_network_update_tau
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
            target_critic_params,
            opt_policy_state,
            opt_critic_state,
            opt_ent_coef_state,
            log_ent_coef,
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

    def _actor_loss(self, policy_params, critic_params, obses, key, ent_coef, diagnostics):
        policy, log_prob, log_std = self._get_pi_log_prob(policy_params, obses, key)
        q1_pi, q2_pi = self.critic(critic_params, policy_params, key, obses, policy)
        actor_loss = jnp.mean(ent_coef * log_prob - jnp.minimum(q1_pi, q2_pi))
        metrics = (
            stochastic_actor_metrics(
                log_prob, log_std, jnp.minimum(q1_pi, q2_pi), ent_coef, self.target_entropy
            )
            if diagnostics
            else {}
        )
        return actor_loss, (log_prob, metrics)

    def _target(
        self,
        policy_params,
        target_critic_params,
        rewards,
        nxtobses,
        not_terminateds,
        key,
        ent_coef,
    ):
        policy, log_prob, _ = self._get_pi_log_prob(policy_params, nxtobses, key)
        q1_pi, q2_pi = self.critic(target_critic_params, policy_params, key, nxtobses, policy)
        next_q = jnp.minimum(q1_pi, q2_pi) - ent_coef * log_prob
        return (not_terminateds * next_q * self._gamma) + rewards
