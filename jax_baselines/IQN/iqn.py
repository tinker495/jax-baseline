from copy import deepcopy

import jax
import jax.numpy as jnp
import optax

from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.losses import QuantileHuberLosses
from jax_baselines.math.metrics import (
    array_metrics,
    quantile_metrics,
    replay_metrics,
    td_metrics,
)
from jax_baselines.math.param_updates import hard_update
from jax_baselines.math.policy_math import q_log_pi
from jax_baselines.optim import optimizer_metrics


class IQN(Q_Network_Family):
    _run_name = "IQN"
    supports_bulk_training = True
    _uses_rng = True

    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        n_support=32,
        delta=1.0,
        CVaR=1.0,
        **kwargs,
    ):
        self.n_support = n_support
        self.delta = delta
        self.CVaR = CVaR
        self.risk_avoid = CVaR != 1.0

        super().__init__(env_builder, model_builder_maker, **kwargs)

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.dueling_model,
            self.param_noise,
            self.policy_kwargs,
        )
        self.preproc, self.model, self.params = model_builder(next(self.key_seq), print_model=True)
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)

        # Use common JIT compilation
        self._compile_common_functions()

    def get_q(self, params, obses, tau, key=None) -> jnp.ndarray:
        return self.model(params, key, self.preproc(params, key, obses), tau)

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        conv_obses = convert_normalized_obs(obses)
        batch_size = next(iter(conv_obses.values())).shape[0]
        tau = jax.random.uniform(key, (batch_size, self.n_support)) * self.CVaR
        return jnp.expand_dims(
            jnp.argmax(
                jnp.mean(self.get_q(params, conv_obses, tau, key), axis=2),
                axis=1,
            ),
            axis=1,
        )

    def _train_step(
        self,
        params,
        target_params,
        opt_state,
        steps,
        key,
        obses,
        actions,
        rewards,
        nxtobses,
        terminateds,
        weights=1,
        *,
        diagnostics,
    ):
        obses = convert_normalized_obs(obses)
        nxtobses = convert_normalized_obs(nxtobses)
        actions = jnp.expand_dims(actions.astype(jnp.int32), axis=2)
        not_terminateds = 1.0 - terminateds
        key1, key2 = jax.random.split(key, 2)
        targets = self._target(
            params,
            target_params,
            obses,
            actions,
            rewards,
            nxtobses,
            not_terminateds,
            key1,
        )
        (loss, (abs_error, quantiles, tau)), grad = jax.value_and_grad(self._loss, has_aux=True)(
            params, obses, actions, targets, weights, key2
        )
        updates, opt_state = self.optimizer.update(
            grad, opt_state, params=params, diagnostics=diagnostics
        )
        q_values = jnp.mean(quantiles, axis=1)
        target_values = jnp.mean(targets, axis=1)
        metrics = (
            {
                **array_metrics(q_values, "loss/q"),
                **array_metrics(target_values, "loss/target"),
                **td_metrics(q_values, target_values),
                **quantile_metrics(quantiles, taus=tau),
                **optimizer_metrics(opt_state, "q"),
                "loss/unweighted_loss": jnp.mean(abs_error),
                "loss/target_stds": jnp.mean(jnp.std(targets, axis=1)),
            }
            if diagnostics
            else {}
        )
        params = optax.apply_updates(params, updates)
        target_params = hard_update(params, target_params, steps, self.target_network_update_freq)
        new_priorities = None
        if self.prioritized_replay:
            new_priorities = abs_error
            if diagnostics:
                metrics.update(replay_metrics(weights, new_priorities))
        return (
            params,
            target_params,
            opt_state,
            loss,
            jnp.mean(targets),
            new_priorities,
            metrics,
            {},
        )

    def _loss(self, params, obses, actions, targets, weights, key):
        tau = jax.random.uniform(key, (self.batch_size, self.n_support))
        theta_loss_tile = jnp.take_along_axis(
            self.get_q(params, obses, tau, key), actions, axis=1
        )  # batch x 1 x support
        logit_valid_tile = jnp.expand_dims(targets, axis=2)  # batch x support x 1
        loss = QuantileHuberLosses(
            logit_valid_tile, theta_loss_tile, jnp.expand_dims(tau, axis=1), self.delta
        )
        return jnp.mean(loss * weights), (loss, theta_loss_tile[:, 0], tau)

    def _target(
        self,
        params,
        target_params,
        obses,
        actions,
        rewards,
        nxtobses,
        not_terminateds,
        key,
    ):
        target_tau = jax.random.uniform(key, (self.batch_size, self.n_support))
        next_q = self.get_q(target_params, nxtobses, target_tau, key)
        action_params = params if self.double_q else target_params
        next_action_q = (
            self.get_q(action_params, nxtobses, target_tau, key) if self.double_q else next_q
        )

        if self.munchausen:
            next_sub_q, tau_log_pi_next = q_log_pi(
                jnp.mean(next_action_q, axis=2), self.munchausen_entropy_tau
            )
            pi_next = jnp.expand_dims(
                jax.nn.softmax(next_sub_q / self.munchausen_entropy_tau), axis=2
            )  # batch x actions x 1
            next_vals = next_q - jnp.expand_dims(
                tau_log_pi_next, axis=2
            )  # batch x actions x support
            next_vals = jnp.sum(pi_next * next_vals, axis=1)

            q_k_targets = jnp.mean(self.get_q(action_params, obses, target_tau, key), axis=2)
            _, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau, clip=True)
            munchausen_addon = jnp.take_along_axis(tau_log_pi, jnp.squeeze(actions, axis=2), axis=1)

            rewards = rewards + self.munchausen_alpha * munchausen_addon
        else:
            next_actions = jnp.argmax(
                jnp.mean(next_action_q, axis=2, keepdims=True), axis=1, keepdims=True
            )
            next_vals = jnp.squeeze(
                jnp.take_along_axis(next_q, next_actions, axis=1)
            )  # batch x support
        return (not_terminateds * next_vals * self._gamma) + rewards  # batch x support

    def run_name_update(self, run_name):
        suffix = (
            f"({self.n_support:d})_CVaR({self.CVaR:.2f})"
            if self.risk_avoid
            else f"({self.n_support:d})"
        )
        return super().run_name_update(run_name + suffix)
