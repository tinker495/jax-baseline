from copy import deepcopy

import jax
import jax.numpy as jnp
import optax

from jax_baselines.common.utils import convert_jax, hard_update, q_log_pi
from jax_baselines.DQN.base_class import Q_Network_Family


class DQN(Q_Network_Family):
    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        num_workers=1,
        eval_eps=20,
        gamma=0.995,
        learning_rate=3e-4,
        buffer_size=100000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        exploration_initial_eps=1.0,
        train_freq=1,
        gradient_steps=1,
        batch_size=32,
        double_q=False,
        dueling_model=False,
        n_step=1,
        learning_starts=1000,
        target_network_update_freq=2000,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-3,
        param_noise=False,
        munchausen=False,
        log_interval=200,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
        compress_memory=False,
    ):
        super().__init__(
            env_builder,
            model_builder_maker,
            num_workers,
            eval_eps,
            gamma,
            learning_rate,
            buffer_size,
            exploration_fraction,
            exploration_final_eps,
            exploration_initial_eps,
            train_freq,
            gradient_steps,
            batch_size,
            double_q,
            dueling_model,
            n_step,
            learning_starts,
            target_network_update_freq,
            prioritized_replay,
            prioritized_replay_alpha,
            prioritized_replay_beta0,
            prioritized_replay_eps,
            param_noise,
            munchausen,
            log_interval,
            log_dir,
            _init_setup_model,
            policy_kwargs,
            full_tensorboard_log,
            seed,
            optimizer,
            compress_memory,
        )

        self.name = "DQN"

        if _init_setup_model:
            self.setup_model()

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

    def get_q(self, params, obses, key=None) -> jnp.ndarray:
        return self.model(params, key, self.preproc(params, key, obses))

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        return jnp.expand_dims(
            jnp.argmax(self.get_q(params, convert_jax(obses), key), axis=1), axis=1
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
            next(self.key_seq) if self.param_noise else None,
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
        actions = actions.astype(jnp.int32)
        not_terminateds = 1.0 - terminateds
        targets = self._target(
            params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key
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
        # loss = hubberloss(error, delta=1.0)
        return jnp.mean(loss * weights), jnp.abs(
            error
        )  # remove weight multiply cpprb weight is something wrong

    def _target(
        self, params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key
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
            next_vals = (
                jnp.sum(pi_next * (next_q - tau_log_pi_next), axis=1, keepdims=True)
                * not_terminateds
            )

            if self.double_q:
                q_k_targets = self.get_q(params, obses, key)
            else:
                q_k_targets = self.get_q(target_params, obses, key)
            _, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau)
            munchausen_addon = jnp.take_along_axis(tau_log_pi, actions, axis=1)

            rewards = rewards + self.munchausen_alpha * jnp.clip(
                munchausen_addon, a_min=-1, a_max=0
            )
        else:
            if self.double_q:
                next_actions = jnp.argmax(self.get_q(params, nxtobses, key), axis=1, keepdims=True)
            else:
                next_actions = jnp.argmax(next_q, axis=1, keepdims=True)
            next_vals = not_terminateds * jnp.take_along_axis(next_q, next_actions, axis=1)
        return (next_vals * self._gamma) + rewards

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="DQN",
        run_name="DQN",
    ):
        super().learn(total_timesteps, callback, log_interval, experiment_name, run_name)
