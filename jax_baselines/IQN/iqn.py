from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.common.losses import QuantileHuberLosses
from jax_baselines.common.utils import convert_jax, hard_update, q_log_pi
from jax_baselines.DQN.base_class import Q_Network_Family


class IQN(Q_Network_Family):
    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        n_support=32,
        delta=1.0,
        CVaR=1.0,
        **kwargs
    ):

        self.name = "IQN"
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

        self.tile_n = self.n_support

        # Use common JIT compilation
        self._compile_common_functions()

    def get_q(self, params, obses, tau, key=None) -> jnp.ndarray:
        return self.model(params, key, self.preproc(params, key, obses), tau)

    def actions(self, obs, epsilon, eval_mode=False):
        if epsilon <= np.random.uniform(0, 1):
            actions = np.asarray(self._get_actions(self.params, obs, next(self.key_seq)))
        else:
            actions = np.random.choice(self.action_size[0], [self.worker_size, 1])
        return actions

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        tau = jax.random.uniform(key, (self.worker_size, self.n_support)) * self.CVaR
        return jnp.expand_dims(
            jnp.argmax(
                jnp.mean(self.get_q(params, convert_jax(obses), tau, key), axis=2),
                axis=1,
            ),
            axis=1,
        )

    def train_step(self, steps, gradient_steps):
        # Use common training step wrapper
        return self._common_train_step_wrapper(steps, gradient_steps, self._train_step_internal)

    def _train_step_internal(self, data):
        """Internal training step that returns the result tuple for the wrapper."""
        return self._train_step(
            self.params,
            self.target_params,
            self.opt_state,
            self.train_steps_count,
            next(self.key_seq),
            **data
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
        indexes=None,
    ):
        obses = convert_jax(obses)
        nxtobses = convert_jax(nxtobses)
        actions = jnp.expand_dims(actions.astype(jnp.int32), axis=2)
        not_terminateds = 1.0 - terminateds
        key1, key2 = jax.random.split(key, 2)
        targets = self._target(
            params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key1
        )
        (loss, abs_error), grad = jax.value_and_grad(self._loss, has_aux=True)(
            params, obses, actions, targets, weights, key2
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
        tau = jax.random.uniform(key, (self.batch_size, self.n_support))
        theta_loss_tile = jnp.take_along_axis(
            self.get_q(params, obses, tau, key), actions, axis=1
        )  # batch x 1 x support
        logit_valid_tile = jnp.expand_dims(targets, axis=2)  # batch x support x 1
        loss = QuantileHuberLosses(
            logit_valid_tile, theta_loss_tile, jnp.expand_dims(tau, axis=1), self.delta
        )
        return jnp.mean(loss * weights), loss

    def _target(
        self, params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key
    ):
        target_tau = jax.random.uniform(key, (self.batch_size, self.n_support))
        next_q = self.get_q(target_params, nxtobses, target_tau, key)

        if self.munchausen:
            if self.double_q:
                next_q_mean = jnp.mean(self.get_q(params, nxtobses, target_tau, key), axis=2)
            else:
                next_q_mean = jnp.mean(next_q, axis=2)
            next_sub_q, tau_log_pi_next = q_log_pi(next_q_mean, self.munchausen_entropy_tau)
            pi_next = jnp.expand_dims(
                jax.nn.softmax(next_sub_q / self.munchausen_entropy_tau), axis=2
            )  # batch x actions x 1
            next_vals = next_q - jnp.expand_dims(
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
                q_k_targets = jnp.mean(self.get_q(params, obses, target_tau, key), axis=2)
            else:
                q_k_targets = jnp.mean(self.get_q(target_params, obses, target_tau, key), axis=2)
            _, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau)
            munchausen_addon = jnp.take_along_axis(tau_log_pi, jnp.squeeze(actions, axis=2), axis=1)

            rewards = rewards + self.munchausen_alpha * jnp.clip(
                munchausen_addon, a_min=-1, a_max=0
            )
        else:
            if self.double_q:
                next_actions = jnp.argmax(
                    jnp.mean(self.get_q(params, nxtobses, target_tau, key), axis=2, keepdims=True),
                    axis=1,
                    keepdims=True,
                )
            else:
                next_actions = jnp.argmax(
                    jnp.mean(next_q, axis=2, keepdims=True), axis=1, keepdims=True
                )
            next_vals = not_terminateds * jnp.squeeze(
                jnp.take_along_axis(next_q, next_actions, axis=1)
            )  # batch x support
        return (next_vals * self._gamma) + rewards  # batch x support

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="IQN",
        run_name="IQN",
    ):
        run_name = run_name + (
            "({:d})_CVaR({:.2f})".format(self.n_support, self.CVaR)
            if self.risk_avoid
            else "({:d})".format(self.n_support)
        )
        super().learn(total_timesteps, callback, log_interval, experiment_name, run_name)
