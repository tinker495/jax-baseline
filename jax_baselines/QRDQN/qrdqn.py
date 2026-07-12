from copy import deepcopy

import jax
import jax.numpy as jnp
import optax

from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.DQN.training import QNetTrainResult
from jax_baselines.math.jax_utils import convert_jax
from jax_baselines.math.losses import QuantileHuberLosses
from jax_baselines.math.param_updates import hard_update
from jax_baselines.math.policy_math import q_log_pi


class QRDQN(Q_Network_Family):
    _run_name = "QRDQN"
    supports_bulk_training = True

    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        n_support=200,
        delta=1.0,
        **kwargs,
    ):
        self.n_support = n_support
        self.delta = delta

        super().__init__(env_builder, model_builder_maker, **kwargs)

    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs

        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.dueling_model,
            self.param_noise,
            self.n_support,
            self.policy_kwargs,
        )
        self.preproc, self.model, self.params = model_builder(next(self.key_seq), print_model=True)
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)

        quantile = (
            jnp.linspace(0.0, 1.0, self.n_support + 1)[1:]
            + jnp.linspace(0.0, 1.0, self.n_support + 1)[:-1]
        ) / 2.0  # [support]
        self.quantile = jax.device_put(
            jnp.tile(jnp.expand_dims(quantile, axis=(0, 1)), (self.batch_size, 1, 1))
        )  # [batch x 1 x support]

        # Use common JIT compilation
        self._compile_common_functions()
        self._bulk_scan = jax.jit(self._bulk_scan)

    def get_q(self, params, obses, key=None) -> jnp.ndarray:
        return self.model(params, key, self.preproc(params, key, obses))

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        return jnp.expand_dims(
            jnp.argmax(jnp.mean(self.get_q(params, convert_jax(obses), key), axis=2), axis=1),
            axis=1,
        )

    def _train_on_batch(self, data, context):
        (
            self.params,
            self.target_params,
            self.opt_state,
            loss,
            t_mean,
            t_std,
            new_priorities,
        ) = self._train_step(
            self.params,
            self.target_params,
            self.opt_state,
            context.train_steps_count,
            next(self.key_seq) if self.param_noise else None,
            **data,
        )
        return QNetTrainResult.from_values(
            loss=loss,
            target=t_mean,
            replay_priorities=new_priorities,
            metrics={"loss/target_stds": t_std},
        )

    def _train_on_bulk(self, data, contexts):
        steps = jnp.asarray([context.train_steps_count for context in contexts])
        keys = jax.random.split(next(self.key_seq), len(contexts)) if self.param_noise else None
        carry = (self.params, self.target_params, self.opt_state)
        (
            (self.params, self.target_params, self.opt_state),
            (
                losses,
                targets,
                target_stds,
                priorities,
            ),
        ) = self._bulk_scan(carry, keys, steps, data)
        return QNetTrainResult.from_values(
            loss=jnp.mean(losses),
            target=jnp.mean(targets),
            replay_priorities=priorities,
            metrics={"loss/target_stds": jnp.mean(target_stds)},
            update_count=len(contexts),
        )

    def _bulk_scan(self, carry, keys, steps, data):
        def train_one(carry, xs):
            params, target_params, opt_state = carry
            if self.param_noise:
                step, key, batch = xs
            else:
                step, batch = xs
                key = None
            params, target_params, opt_state, loss, t_mean, t_std, priorities = self._train_step(
                params,
                target_params,
                opt_state,
                step,
                key,
                **batch,
            )
            return (params, target_params, opt_state), (loss, t_mean, t_std, priorities)

        xs = (steps, keys, data) if self.param_noise else (steps, data)
        return jax.lax.scan(train_one, carry, xs)

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
        indexes=None,
    ):
        obses = convert_jax(obses)
        nxtobses = convert_jax(nxtobses)
        actions = jnp.expand_dims(actions.astype(jnp.int32), axis=2)
        not_terminateds = 1.0 - terminateds
        targets = self._target(
            params,
            target_params,
            obses,
            actions,
            rewards,
            nxtobses,
            not_terminateds,
            key,
        )
        (loss, abs_error), grad = jax.value_and_grad(self._loss, has_aux=True)(
            params, obses, actions, targets, weights, key
        )
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        target_params = hard_update(params, target_params, steps, self.target_network_update_freq)
        new_priorities = None
        if self.prioritized_replay:
            new_priorities = abs_error
        return (
            params,
            target_params,
            opt_state,
            loss,
            jnp.mean(targets),
            jnp.mean(jnp.std(targets, axis=1)),
            new_priorities,
        )

    def _loss(self, params, obses, actions, targets, weights, key):
        theta_loss_tile = jnp.take_along_axis(
            self.get_q(params, obses, key), actions, axis=1
        )  # batch x 1 x support
        logit_valid_tile = jnp.expand_dims(targets, axis=2)  # batch x support x 1
        loss = QuantileHuberLosses(logit_valid_tile, theta_loss_tile, self.quantile, self.delta)
        return jnp.mean(loss * weights), loss

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
        next_q = self.get_q(target_params, nxtobses, key)

        if self.munchausen:
            if self.double_q:
                next_q_mean = jnp.mean(self.get_q(params, nxtobses, key), axis=2)
            else:
                next_q_mean = jnp.mean(next_q, axis=2)
            next_sub_q, tau_log_pi_next = q_log_pi(next_q_mean, self.munchausen_entropy_tau)
            pi_next = jnp.expand_dims(
                jax.nn.softmax(next_sub_q / self.munchausen_entropy_tau), axis=2
            )  # batch x actions x 1
            next_vals = next_q - jnp.expand_dims(
                tau_log_pi_next, axis=2
            )  # batch x actions x support
            next_vals = jnp.sum(pi_next * next_vals, axis=1)

            if self.double_q:
                q_k_targets = jnp.mean(self.get_q(params, obses, key), axis=2)
            else:
                q_k_targets = jnp.mean(self.get_q(target_params, obses, key), axis=2)
            _, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau, clip=True)
            munchausen_addon = jnp.take_along_axis(tau_log_pi, jnp.squeeze(actions, axis=2), axis=1)

            rewards = rewards + self.munchausen_alpha * munchausen_addon
        else:
            if self.double_q:
                next_actions = jnp.argmax(
                    jnp.mean(self.get_q(params, nxtobses, key), axis=2, keepdims=True),
                    axis=1,
                    keepdims=True,
                )
            else:
                next_actions = jnp.argmax(
                    jnp.mean(next_q, axis=2, keepdims=True), axis=1, keepdims=True
                )
            next_vals = jnp.squeeze(
                jnp.take_along_axis(next_q, next_actions, axis=1)
            )  # batch x support
        return (not_terminateds * next_vals * self._gamma) + rewards  # batch x support

    def run_name_update(self, run_name):
        return super().run_name_update(f"{run_name}({self.n_support:d})")
