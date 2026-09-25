from copy import deepcopy

import jax
import jax.numpy as jnp
import optax

from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.losses import FQFQuantileLosses, QuantileHuberLosses
from jax_baselines.math.metrics import (
    array_metrics,
    quantile_metrics,
    replay_metrics,
    td_metrics,
)
from jax_baselines.math.param_updates import hard_update
from jax_baselines.math.policy_math import q_log_pi
from jax_baselines.optim import (
    OptimizerFactory,
    optimizer_metrics,
    require_optimizer_factory,
)


class FQF(Q_Network_Family):
    _run_name = "FQF"
    supports_bulk_training = True
    _uses_rng = True

    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        n_support=32,
        delta=1.0,
        fqf_optimizer_factory: OptimizerFactory | None = None,
        **kwargs,
    ):
        self.fqf_optimizer_factory = require_optimizer_factory(fqf_optimizer_factory)
        self.n_support = n_support
        self.delta = delta
        self.fqf_factor = 1e-2
        self.ent_coef = 0.01

        super().__init__(env_builder, model_builder_maker, **kwargs)

    def _make_fqf_optimizer(self):
        return optax.with_extra_args_support(
            self.fqf_optimizer_factory(self.learning_rate * self.fqf_factor)
        )

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.dueling_model,
            self.param_noise,
            self.n_support,
            self.policy_kwargs,
        )

        self.preproc, self.model, self.fpf, self.params, self.fqf_params = model_builder(
            next(self.key_seq), print_model=True
        )
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)

        self.fqf_optimizer = self._make_fqf_optimizer()
        self.fqf_opt_state = self.fqf_optimizer.init(self.fqf_params)

        # Use common JIT compilation
        self._compile_common_functions()

    @property
    def _train_state(self):
        return (
            self.params,
            self.fqf_params,
            self.target_params,
            self.opt_state,
            self.fqf_opt_state,
        )

    @_train_state.setter
    def _train_state(self, state):
        (
            self.params,
            self.fqf_params,
            self.target_params,
            self.opt_state,
            self.fqf_opt_state,
        ) = state

    def _acting_params(self, params):
        # Acting always reads the live fraction network, including checkpoint evaluation.
        return params, self.fqf_params

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        params, fqf_params = params
        feature = self.preproc(params, key, convert_normalized_obs(obses))
        tau, tau_hat, _ = self.fpf(fqf_params, key, feature)
        return jnp.argmax(self.get_q(params, feature, tau, tau_hat, key), axis=1, keepdims=True)

    def get_quantile(self, params, feature, tau_hat, key=None) -> jnp.ndarray:
        return self.model(params, key, feature, tau_hat)

    def quantiles_to_q(self, quantiles, tau):
        tau = jnp.expand_dims(tau, axis=1)
        q = (tau[:, :, 1:] - tau[:, :, :-1]) * quantiles
        return jnp.sum(q, axis=2)

    def get_q(self, params, feature, tau, tau_hat, key=None) -> jnp.ndarray:
        return self.quantiles_to_q(self.get_quantile(params, feature, tau_hat, key), tau)

    def _train_step(
        self,
        params,
        fqf_params,
        target_params,
        opt_state,
        fqf_opt_state,
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
        (
            (
                loss,
                (
                    abs_error,
                    feature,
                    tau_hats,
                    theta_loss_tile,
                    targets,
                    target_weights,
                    metrics,
                ),
            ),
            grad,
        ) = jax.value_and_grad(self._loss, has_aux=True)(
            params,
            fqf_params,
            target_params,
            obses,
            actions,
            rewards,
            nxtobses,
            not_terminateds,
            weights,
            key,
            diagnostics=diagnostics,
        )
        fqf_loss, grad_fqf = jax.value_and_grad(self._fqf_loss)(
            fqf_params, params, feature, actions, theta_loss_tile, key
        )
        fqf_update, fqf_opt_state = self.fqf_optimizer.update(
            grad_fqf, fqf_opt_state, params=fqf_params, diagnostics=diagnostics
        )
        fqf_params = optax.apply_updates(fqf_params, fqf_update)
        updates, opt_state = self.optimizer.update(
            grad, opt_state, params=params, diagnostics=diagnostics
        )
        if diagnostics:
            metrics.update(
                {
                    **optimizer_metrics(opt_state, "q"),
                    **optimizer_metrics(fqf_opt_state, "fraction"),
                    "loss/fqf_loss": fqf_loss,
                    "loss/target_stds": jnp.mean(jnp.std(targets, axis=1)),
                }
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
            fqf_params,
            target_params,
            opt_state,
            fqf_opt_state,
            loss,
            jnp.mean(jnp.sum(target_weights * targets, axis=1)),
            new_priorities,
            metrics,
            {"loss/tau": tau_hats} if diagnostics else {},
        )

    def _loss(
        self,
        params,
        fqf_params,
        target_params,
        obses,
        actions,
        rewards,
        nxtobses,
        not_terminateds,
        weights,
        key,
        diagnostics,
    ):
        feature = self.preproc(params, key, obses)
        taus, tau_hats, entropy = self.fpf(fqf_params, key, jax.lax.stop_gradient(feature))
        targets, target_weights = jax.lax.stop_gradient(
            self._target(
                params,
                fqf_params,
                target_params,
                taus,
                tau_hats,
                obses,
                actions,
                rewards,
                nxtobses,
                not_terminateds,
                key,
            )
        )
        tau_hats = jax.lax.stop_gradient(tau_hats)
        theta_loss_tile = jnp.take_along_axis(
            self.get_quantile(params, feature, tau_hats, key),
            actions,
            axis=1,
        )  # batch x 1 x support
        logit_valid_tile = jnp.expand_dims(targets, axis=2)  # batch x support x 1
        logit_valid_weight = jnp.expand_dims(target_weights, axis=2)  # batch x support x 1
        hubber = QuantileHuberLosses(
            logit_valid_tile,
            theta_loss_tile,
            jnp.expand_dims(tau_hats, axis=1),
            self.delta,
            logit_valid_weight,
        )
        aux = (hubber, feature, tau_hats, theta_loss_tile, targets, target_weights)
        if not diagnostics:
            return jnp.mean(hubber * weights), (*aux, {})
        widths = taus[:, 1:] - taus[:, :-1]
        q_values = jnp.sum(widths * theta_loss_tile[:, 0], axis=1)
        target_values = jnp.sum(target_weights * targets, axis=1)
        return jnp.mean(hubber * weights), (
            *aux,
            {
                **array_metrics(q_values, "loss/q"),
                **array_metrics(target_values, "loss/target"),
                **td_metrics(q_values, target_values),
                **quantile_metrics(theta_loss_tile[:, 0]),
                "loss/unweighted_loss": jnp.mean(hubber),
                "loss/fraction_entropy": jnp.mean(entropy),
                "loss/fraction_width_min": jnp.min(widths),
                "loss/fraction_width_max": jnp.max(widths),
            },
        )

    def _fqf_loss(self, fqf_params, params, feature, actions, tau_hat_vals, key):
        feature = jax.lax.stop_gradient(feature)
        tau, _, entropy = self.fpf(fqf_params, key, feature)
        tau_vals = jnp.take_along_axis(
            self.get_quantile(params, feature, tau[:, 1:-1], key), actions, axis=1
        )  # batch x 1 x support
        tau_vals = jnp.squeeze(tau_vals)
        tau_hat_vals = jax.lax.stop_gradient(jnp.squeeze(tau_hat_vals))
        quantile_loss = jnp.mean(
            FQFQuantileLosses(
                tau_vals,
                tau_hat_vals,
                tau,
            )
        )
        entropy_loss = -self.ent_coef * jnp.mean(entropy)
        loss = quantile_loss + entropy_loss
        return loss

    def _target(
        self,
        params,
        fqf_params,
        target_params,
        taus,
        tau_hats,
        obses,
        actions,
        rewards,
        nxtobses,
        not_terminateds,
        key,
    ):
        feature = self.preproc(target_params, key, nxtobses)
        online_feature = self.preproc(params, key, nxtobses)
        _tau, _tau_hats, _ = self.fpf(fqf_params, key, online_feature)
        target_weights = _tau[:, 1:] - _tau[:, :-1]
        next_quantiles = self.get_quantile(
            target_params,
            feature,
            _tau_hats,
            key,
        )
        action_params = params if self.double_q else target_params

        if self.double_q:
            next_q = self.get_q(
                action_params,
                online_feature,
                _tau,
                _tau_hats,
                key,
            )
        else:
            next_q = self.quantiles_to_q(next_quantiles, _tau)

        if self.munchausen:
            next_sub_q, tau_log_pi_next = q_log_pi(next_q, self.munchausen_entropy_tau)
            pi_next = jnp.expand_dims(
                jax.nn.softmax(next_sub_q / self.munchausen_entropy_tau), axis=2
            )  # batch x actions x 1
            next_vals = next_quantiles - jnp.expand_dims(
                tau_log_pi_next, axis=2
            )  # batch x actions x support
            next_vals = jnp.sum(pi_next * next_vals, axis=1)

            feature = self.preproc(action_params, key, obses)
            q_k_targets = self.get_q(action_params, feature, taus, tau_hats, key)
            _, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau, clip=True)
            munchausen_addon = jnp.take_along_axis(tau_log_pi, jnp.squeeze(actions, axis=2), axis=1)

            rewards = rewards + self.munchausen_alpha * munchausen_addon
        else:
            next_actions = jnp.expand_dims(jnp.argmax(next_q, axis=1), axis=(1, 2))
            next_vals = jnp.squeeze(
                jnp.take_along_axis(next_quantiles, next_actions, axis=1)
            )  # batch x support
        return (not_terminateds * next_vals * self._gamma) + rewards, target_weights

    def run_name_update(self, run_name):
        return super().run_name_update(f"{run_name}({self.n_support:d})")
