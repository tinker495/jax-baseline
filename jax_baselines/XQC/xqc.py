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
from jax_baselines.math.distributional import categorical_projection
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.metrics import categorical_metrics, support_metrics
from jax_baselines.math.param_updates import (
    project_dense_kernels,
    scaled_by_reset,
    soft_update,
)
from jax_baselines.math.policy_math import entropy_target_from_sigma
from jax_baselines.optim import optimizer_metrics


@struct.dataclass
class XQCCheckpointParams:
    policy_params: Any
    critic_params: Any
    target_critic_params: Any
    log_ent_coef: Any


class XQC(Deteministic_Policy_Gradient_Family):
    _run_name = "XQC"

    @property
    def _actor_schedule(self):
        return self.policy_delay, 1

    def __init__(
        self,
        env_builder: Callable,
        model_builder_maker,
        ent_coef="auto_0.01",
        sigma_target=0.15,
        policy_delay=3,
        ent_coef_learning_rate=3e-4,
        lr_end=3e-5,
        lr_transition_steps=2_000_000,
        value_min=-5.0,
        value_max=5.0,
        n_atoms=101,
        reward_normalization=True,
        **kwargs,
    ):
        xqc_kwargs = {
            "target_network_update_tau": 0.005,
            "reward_normalization": reward_normalization,
            **kwargs,
        }

        self._ent_coef = ent_coef
        self.lr_end = lr_end
        self.lr_transition_steps = lr_transition_steps
        self.ent_coef_learning_rate = optax.linear_schedule(
            ent_coef_learning_rate, lr_end, lr_transition_steps
        )
        self.policy_delay = policy_delay
        self.value_min = value_min
        self.value_max = value_max
        self.n_atoms = n_atoms
        # Host constants: a captured device array is copied back to the host at every compile.
        self.value_support = np.asarray(jnp.linspace(value_min, value_max, n_atoms))
        self.support_delta = (value_max - value_min) / (n_atoms - 1)

        super().__init__(env_builder, model_builder_maker, **xqc_kwargs)

        self.target_entropy = entropy_target_from_sigma(
            int(np.prod(self.action_size)), sigma_target
        )

    def _make_optimizer(self, learning_rate):
        schedule = optax.linear_schedule(learning_rate, self.lr_end, self.lr_transition_steps)
        return self.optimizer_factory(schedule)

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
        self.policy_params = project_dense_kernels(self.policy_params)
        self.critic_params = project_dense_kernels(self.critic_params)
        self.target_critic_params = deepcopy(self.critic_params)
        self.opt_policy_state = self.optimizer.init(self.policy_params)
        self.opt_critic_state = self.optimizer.init(self.critic_params)

        self._setup_entropy_coef()

    def checkpoint_params(self):
        return XQCCheckpointParams(
            policy_params=self.policy_params,
            critic_params=self.critic_params,
            target_critic_params=self.target_critic_params,
            log_ent_coef=self.log_ent_coef,
        )

    def load_checkpoint_params(self, bundle):
        self.policy_params = bundle.policy_params
        self.critic_params = bundle.critic_params
        self.target_critic_params = getattr(bundle, "target_critic_params", bundle.critic_params)
        self.log_ent_coef = bundle.log_ent_coef

    def _get_pi_log_prob(self, params, obses, key, training: bool = True):
        (mu, log_std), updates = self.actor(params, None, obses, training)
        params["batch_stats"] = updates["batch_stats"]
        std = jnp.exp(log_std)
        x_t = mu + std * jax.random.normal(key, std.shape)
        pi = jax.nn.tanh(x_t)
        log_prob = jnp.sum(
            -0.5 * (jnp.square((x_t - mu) / (std + 1e-6)) + 2 * log_std + jnp.log(2 * np.pi))
            - jnp.log(1 - jnp.square(pi) + 1e-6),
            axis=1,
            keepdims=True,
        )
        return pi, log_prob, params, log_std

    def _get_actions(self, state, obses, key):
        key, sample_key = jax.random.split(key)
        (mu, log_std), _ = self.actor(state["policy"], None, convert_normalized_obs(obses), False)
        std = jnp.exp(log_std)
        return jax.nn.tanh(mu + std * jax.random.normal(sample_key, std.shape)), key

    def _get_eval_actions(self, state, obses):
        (mu, _), _ = self.actor(state["policy"], None, convert_normalized_obs(obses), False)
        return jax.nn.tanh(mu)

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
        key1, key2 = jax.random.split(key, 2)
        target_distribution, target_q, next_policy, target_metrics = self._target(
            policy_params,
            target_critic_params,
            obses,
            actions,
            rewards,
            nxtobses,
            not_terminateds,
            ent_coef,
            key1,
            flags.diagnostics,
        )

        (critic_loss, (cross_entropy, critic_params, metrics)), grad = jax.value_and_grad(
            self._critic_loss, has_aux=True
        )(
            critic_params,
            policy_params,
            obses,
            actions,
            nxtobses,
            next_policy,
            target_distribution,
            weights,
            key1,
            flags.diagnostics,
        )
        updates, opt_critic_state = self.optimizer.update(
            grad, opt_critic_state, params=critic_params, diagnostics=flags.diagnostics
        )
        critic_params = optax.apply_updates(critic_params, updates)
        critic_params = project_dense_kernels(critic_params)
        target_critic_params = soft_update(
            critic_params, target_critic_params, self.target_network_update_tau
        )

        actor_metrics = {}
        if flags.actor:
            (actor_loss, (log_prob, policy_params, actor_metrics)), grad = jax.value_and_grad(
                self._actor_loss, has_aux=True
            )(policy_params, critic_params, obses, key2, ent_coef, flags.diagnostics)
            updates, opt_policy_state = self.optimizer.update(
                grad, opt_policy_state, params=policy_params, diagnostics=flags.diagnostics
            )
            policy_params = optax.apply_updates(policy_params, updates)
            policy_params = project_dense_kernels(policy_params)
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
            metrics.update(target_metrics)
            metrics["loss/ent_coef"] = jnp.exp(log_ent_coef)
            metrics["loss/targets"] = jnp.mean(target_q)
        metrics, metric_counts = merge_actor_metrics(metrics, actor_metrics, flags.actor)

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
        ), (cross_entropy if self.prioritized_replay else None, metrics, metric_counts)

    def _critic_loss(
        self,
        critic_params,
        policy_params,
        obses,
        actions,
        nxtobses,
        next_policy,
        target_distribution,
        weights,
        key,
        diagnostics,
    ):
        concated_obses = {
            name: jnp.concatenate([obs, nxtobses[name]]) for name, obs in obses.items()
        }
        concated_actions = jnp.concatenate([actions, next_policy])
        (logits1, logits2), variable_updates = self.critic(
            critic_params, policy_params, key, concated_obses, concated_actions, True
        )
        critic_params = {**critic_params, **variable_updates}
        logits1 = jnp.split(logits1, 2, axis=0)[0]
        logits2 = jnp.split(logits2, 2, axis=0)[0]
        cross_entropy1 = -jnp.sum(
            target_distribution * jax.nn.log_softmax(logits1, axis=-1), axis=-1
        )
        cross_entropy2 = -jnp.sum(
            target_distribution * jax.nn.log_softmax(logits2, axis=-1), axis=-1
        )
        weights = jnp.asarray(weights).squeeze()
        critic_loss = jnp.mean(weights * cross_entropy1) + jnp.mean(weights * cross_entropy2)
        cross_entropy = 0.5 * (cross_entropy1 + cross_entropy2)
        if not diagnostics:
            return critic_loss, (cross_entropy, critic_params, {})
        metrics = critic_metrics(
            (self._categorical_q(logits1), self._categorical_q(logits2)),
            jnp.sum(target_distribution * self.value_support, axis=-1),
            (cross_entropy1, cross_entropy2),
            weights,
            cross_entropy if self.prioritized_replay else None,
        )
        for index, logits in enumerate((logits1, logits2), start=1):
            metrics.update(
                categorical_metrics(
                    jax.nn.softmax(logits, axis=-1),
                    target_distribution,
                    self.value_support,
                    prefix=f"loss/critic{index}",
                )
            )
        return critic_loss, (cross_entropy, critic_params, metrics)

    def _categorical_q(self, logits):
        return jnp.sum(jax.nn.softmax(logits, axis=-1) * self.value_support, axis=-1)

    def _actor_loss(self, policy_params, critic_params, obses, key, ent_coef, diagnostics):
        policy, log_prob, policy_params, log_std = self._get_pi_log_prob(policy_params, obses, key)
        (logits1, logits2), _ = self.critic(critic_params, policy_params, key, obses, policy, False)
        q1_pi = self._categorical_q(logits1)
        q2_pi = self._categorical_q(logits2)
        actor_loss = jnp.mean(ent_coef * jnp.squeeze(log_prob, axis=-1) - jnp.minimum(q1_pi, q2_pi))
        if not diagnostics:
            return actor_loss, (log_prob, policy_params, {})
        return actor_loss, (
            log_prob,
            policy_params,
            stochastic_actor_metrics(
                log_prob,
                log_std,
                jnp.minimum(q1_pi, q2_pi),
                ent_coef,
                self.target_entropy,
            ),
        )

    def _target(
        self,
        policy_params,
        target_critic_params,
        obses,
        actions,
        rewards,
        nxtobses,
        not_terminateds,
        ent_coef,
        key,
        diagnostics,
    ):
        concated_obses = {
            name: jnp.concatenate([obs, nxtobses[name]]) for name, obs in obses.items()
        }
        next_policy, log_prob, _, _ = self._get_pi_log_prob(policy_params, nxtobses, key, False)
        concated_actions = jnp.concatenate([actions, next_policy])
        (logits1, logits2), _ = self.critic(
            target_critic_params,
            policy_params,
            key,
            concated_obses,
            concated_actions,
            True,
        )
        next_logits1 = jnp.split(logits1, 2, axis=0)[1]
        next_logits2 = jnp.split(logits2, 2, axis=0)[1]
        next_probs1 = jax.nn.softmax(next_logits1, axis=-1)
        next_probs2 = jax.nn.softmax(next_logits2, axis=-1)
        choose_first = self._categorical_q(next_logits1) <= self._categorical_q(next_logits2)
        next_probs = jnp.where(choose_first[:, None], next_probs1, next_probs2)
        shifted_atoms = rewards.reshape(-1, 1) + (
            not_terminateds.reshape(-1, 1)
            * self._gamma
            * (self.value_support[None, :] - ent_coef * log_prob.reshape(-1, 1))
        )
        target_distribution = categorical_projection(
            next_probs,
            shifted_atoms,
            self.value_min,
            self.value_max,
            self.support_delta,
            self.n_atoms,
        )
        target_distribution = jax.lax.stop_gradient(target_distribution)
        target_q = jnp.sum(target_distribution * self.value_support, axis=-1)
        return (
            target_distribution,
            target_q,
            next_policy,
            (
                support_metrics(next_probs, shifted_atoms, self.value_min, self.value_max)
                if diagnostics
                else {}
            ),
        )
