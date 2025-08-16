from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.common.losses import FQFQuantileLosses, QuantileHuberLosses
from jax_baselines.common.optimizer import select_optimizer
from jax_baselines.common.utils import convert_jax, hard_update, q_log_pi
from jax_baselines.DQN.base_class import Q_Network_Family


class FQF(Q_Network_Family):
    def __init__(
        self, env_builder: callable, model_builder_maker, n_support=32, delta=1.0, **kwargs
    ):
        super().__init__(env_builder, model_builder_maker, **kwargs)

        self.name = "FQF"
        self.n_support = n_support
        self.delta = delta
        self.fqf_factor = 1e-2
        self.ent_coef = 0.01

        if kwargs.get("_init_setup_model", True):
            self.setup_model()

    def setup_model(self):
        self.model_bulder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.dueling_model,
            self.param_noise,
            self.n_support,
            self.policy_kwargs,
        )

        self.preproc, self.model, self.fpf, self.params, self.fqf_params = self.model_bulder(
            next(self.key_seq), print_model=True
        )
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)

        self.fqf_optimizer = select_optimizer(
            "rmsprop", self.learning_rate * self.fqf_factor, grad_max=5.0
        )
        self.fqf_opt_state = self.fqf_optimizer.init(self.fqf_params)

        # Use common JIT compilation
        self._compile_common_functions()

    def actions(self, obs, epsilon):
        if epsilon <= np.random.uniform(0, 1):
            actions = np.asarray(
                self._get_actions(
                    self.params,
                    self.fqf_params,
                    obs,
                    next(self.key_seq) if self.param_noise else None,
                )
            )
        else:
            actions = np.random.choice(self.action_size[0], [self.worker_size, 1])
        return actions

    def _get_actions(self, params, fqf_params, obses, key=None) -> jnp.ndarray:
        feature = self.preproc(params, key, convert_jax(obses))
        tau, tau_hat, _ = self.fpf(fqf_params, key, feature)
        return jnp.argmax(self.get_q(params, feature, tau, tau_hat, key), axis=1, keepdims=True)

    def get_quantile(self, params, feature, tau_hat, key=None) -> jnp.ndarray:
        return self.model(params, key, feature, tau_hat)

    def quantiles_to_q(self, quantiles, tau):
        tau = jnp.expand_dims(tau, axis=1)
        q = (tau[:, :, 1:] - tau[:, :, :-1]) * quantiles
        return jnp.sum(q, axis=2)

    def get_q(self, params, feature, tau, tau_hat, key=None) -> jnp.ndarray:
        quanile_hat = self.model(params, key, feature, tau_hat)
        tau = jnp.expand_dims(tau, axis=1)
        q = (tau[:, :, 1:] - tau[:, :, :-1]) * quanile_hat
        return jnp.sum(q, axis=2)

    def train_step(self, steps, gradient_steps):
        # FQF has a more complex structure, so we handle it specially
        for _ in range(gradient_steps):
            self.train_steps_count += 1
            data = self._sample_batch()

            (
                self.params,
                self.fqf_params,
                self.target_params,
                self.opt_state,
                self.fqf_opt_state,
                loss,
                fqf_loss,
                t_mean,
                t_std,
                tau,
                new_priorities,
            ) = self._train_step(
                self.params,
                self.fqf_params,
                self.target_params,
                self.opt_state,
                self.fqf_opt_state,
                self.train_steps_count,
                next(self.key_seq),
                **data
            )

            self._update_priorities(data, new_priorities)

        if self.logger_run and steps % self.log_interval == 0:
            self.logger_run.log_metric("loss/qloss", loss, steps)
            self.logger_run.log_metric("loss/fqf_loss", fqf_loss, steps)
            self.logger_run.log_metric("loss/targets", t_mean, steps)
            self.logger_run.log_metric("loss/target_stds", t_std, steps)
            self.logger_run.log_histogram("loss/tau", tau, steps)

        return loss

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
        indexes=None,
    ):
        obses = convert_jax(obses)
        nxtobses = convert_jax(nxtobses)
        actions = jnp.expand_dims(actions.astype(jnp.int32), axis=2)
        not_terminateds = 1.0 - terminateds
        (
            loss,
            (abs_error, feature, taus, tau_hats, theta_loss_tile, targets, target_weights),
        ), grad = jax.value_and_grad(self._loss, has_aux=True,)(
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
        )
        fqf_loss, grad_fqf = jax.value_and_grad(self._fqf_loss)(
            fqf_params, params, feature, actions, theta_loss_tile, key
        )
        fqf_update, fqf_opt_state = self.fqf_optimizer.update(
            grad_fqf, fqf_opt_state, params=fqf_params
        )
        fqf_params = optax.apply_updates(fqf_params, fqf_update)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        target_params = hard_update(params, target_params, steps, self.target_network_update_freq)
        new_priorities = None
        if self.prioritized_replay:
            new_priorities = abs_error
        return (
            params,
            fqf_params,
            target_params,
            opt_state,
            fqf_opt_state,
            loss,
            fqf_loss,
            jnp.mean(jnp.sum(target_weights * targets, axis=1)),
            jnp.mean(jnp.std(targets, axis=1)),
            tau_hats,
            new_priorities,
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
    ):
        feature = self.preproc(params, key, obses)
        taus, tau_hats, _ = self.fpf(fqf_params, key, jax.lax.stop_gradient(feature))
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
        return jnp.mean(hubber * weights), (
            hubber,
            feature,
            taus,
            tau_hats,
            theta_loss_tile,
            targets,
            target_weights,
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

        if self.double_q:
            next_q = self.get_q(
                params,
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
            sample_pi = jax.random.categorical(
                key, jnp.tile(pi_next, (1, 1, self.n_support)), 1
            )  # batch x 1 x support
            next_vals = jnp.take_along_axis(
                next_vals, jnp.expand_dims(sample_pi, axis=1), axis=1
            )  # batch x 1 x support
            next_vals = not_terminateds * jnp.squeeze(next_vals, axis=1)

            if self.double_q:
                feature = self.preproc(params, key, obses)
                q_k_targets = self.get_q(params, feature, taus, tau_hats, key)
            else:
                feature = self.preproc(target_params, key, obses)
                q_k_targets = self.get_q(target_params, feature, taus, tau_hats, key)
            _, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau)
            munchausen_addon = jnp.take_along_axis(tau_log_pi, jnp.squeeze(actions, axis=2), axis=1)

            rewards = rewards + self.munchausen_alpha * jnp.clip(
                munchausen_addon, a_min=-1, a_max=0
            )
        else:
            next_actions = jnp.expand_dims(jnp.argmax(next_q, axis=1), axis=(1, 2))
            next_vals = not_terminateds * jnp.squeeze(
                jnp.take_along_axis(next_quantiles, next_actions, axis=1)
            )  # batch x support
        return (next_vals * self._gamma) + rewards, target_weights

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="FQF",
        run_name="FQF",
    ):
        run_name = run_name + "({:d})".format(self.n_support)
        super().learn(total_timesteps, callback, log_interval, experiment_name, run_name)
