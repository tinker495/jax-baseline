from copy import deepcopy

import jax
import jax.numpy as jnp
import optax

from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.math.jax_utils import convert_jax
from jax_baselines.math.param_updates import hard_update
from jax_baselines.math.policy_math import q_log_pi


class DQN(Q_Network_Family):
    supports_bulk_training = True

    def __init__(self, env_builder: callable, model_builder_maker, **kwargs):
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
        self._bulk_scan = jax.jit(self._bulk_scan)

    def get_q(self, params, obses, key=None) -> jnp.ndarray:
        return self.model(params, key, self.preproc(params, key, obses))

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        return jnp.expand_dims(
            jnp.argmax(self.get_q(params, convert_jax(obses), key), axis=1), axis=1
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
        actions = actions.astype(jnp.int32)
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
        return params, target_params, opt_state, loss, jnp.mean(targets), new_priorities

    def _loss(self, params, obses, actions, targets, weights, key):
        vals = jnp.take_along_axis(self.get_q(params, obses, key), actions, axis=1)
        error = jnp.squeeze(vals - targets)
        loss = jnp.square(error)
        return jnp.mean(loss * weights), jnp.abs(
            error
        )  # remove weight multiply cpprb weight is something wrong

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
                next_sub_q, tau_log_pi_next = q_log_pi(
                    self.get_q(params, nxtobses, key), self.munchausen_entropy_tau
                )
            else:
                next_sub_q, tau_log_pi_next = q_log_pi(next_q, self.munchausen_entropy_tau)
            pi_next = jax.nn.softmax(next_sub_q / self.munchausen_entropy_tau)
            next_vals = jnp.sum(pi_next * (next_q - tau_log_pi_next), axis=1, keepdims=True)

            if self.double_q:
                q_k_targets = self.get_q(params, obses, key)
            else:
                q_k_targets = self.get_q(target_params, obses, key)
            _, clipped_tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau, clip=True)
            munchausen_addon = jnp.take_along_axis(clipped_tau_log_pi, actions, axis=1)

            rewards = rewards + self.munchausen_alpha * munchausen_addon
        else:
            if self.double_q:
                next_actions = jnp.argmax(self.get_q(params, nxtobses, key), axis=1, keepdims=True)
            else:
                next_actions = jnp.argmax(next_q, axis=1, keepdims=True)
            next_vals = jnp.take_along_axis(next_q, next_actions, axis=1)
        return (not_terminateds * next_vals * self._gamma) + rewards

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="DQN",
        run_name="DQN",
        eval_num=100,
        logger_factory=None,
        progress_factory=None,
        record_test_fn=None,
    ):
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            experiment_name,
            run_name,
            eval_num,
            logger_factory=logger_factory,
            progress_factory=progress_factory,
            record_test_fn=record_test_fn,
        )
