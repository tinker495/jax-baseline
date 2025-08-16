from copy import deepcopy

import jax
import jax.numpy as jnp
import optax

from jax_baselines.common.utils import convert_jax, hard_update, q_log_pi
from jax_baselines.DQN.base_class import Q_Network_Family


class HL_GAUSS_C51(Q_Network_Family):
    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        categorial_bar_n=51,
        categorial_max=250,
        categorial_min=-250,
        **kwargs
    ):
        super().__init__(env_builder, model_builder_maker, **kwargs)

        self.name = "HL_GAUSS_C51"
        self.sigma = 0.75
        self.categorial_bar_n = categorial_bar_n
        self.categorial_max = float(categorial_max)
        self.categorial_min = float(categorial_min)

    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs

        self.model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.dueling_model,
            self.param_noise,
            self.categorial_bar_n,
            self.policy_kwargs,
        )

        self.preproc, self.model, self.params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)

        self.support = jnp.linspace(
            self.categorial_min, self.categorial_max, self.categorial_bar_n + 1, dtype=jnp.float32
        )
        bin_width = self.support[1] - self.support[0]
        self.sigma = self.sigma * bin_width

        # Use common JIT compilation
        self._compile_common_functions()

    def get_q(self, params, obses, key=None) -> jnp.ndarray:
        return self.model(params, key, self.preproc(params, key, obses))

    def to_probs(self, target: jax.Array):
        # target: [batch, 1]
        def f(target):
            cdf_evals = jax.scipy.special.erf((self.support - target) / (jnp.sqrt(2) * self.sigma))
            z = cdf_evals[-1] - cdf_evals[0]
            bin_probs = cdf_evals[1:] - cdf_evals[:-1]
            return bin_probs / z

        return jax.vmap(f)(target)

    def to_scalar(self, probs: jax.Array):
        # probs: [batch, n, support]
        def f(probs):
            centers = (self.support[:-1] + self.support[1:]) / 2
            return jnp.sum(probs * centers)

        return jax.vmap(jax.vmap(f))(probs)

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        return jnp.argmax(
            self.to_scalar(self.get_q(params, convert_jax(obses), key)), axis=1, keepdims=True
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
        actions = jnp.expand_dims(actions.astype(jnp.int32), axis=2)
        not_terminateds = 1.0 - terminateds
        target_distribution = self._target(
            params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key
        )
        (loss, centropy), grad = jax.value_and_grad(self._loss, has_aux=True)(
            params, obses, actions, target_distribution, weights, key
        )
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        target_params = hard_update(params, target_params, steps, self.target_network_update_freq)
        new_priorities = None
        if self.prioritized_replay:
            new_priorities = centropy
        return (
            params,
            target_params,
            opt_state,
            loss,
            self.to_scalar(jnp.expand_dims(target_distribution, 1)).mean(),
            new_priorities,
        )

    def _loss(self, params, obses, actions, target_distribution, weights, key):
        distribution = jnp.squeeze(
            jnp.take_along_axis(self.get_q(params, obses, key), actions, axis=1)
        )
        cross_entropy = -jnp.sum(target_distribution * jnp.log(distribution + 1e-6), axis=1)
        return jnp.mean(cross_entropy), cross_entropy

    def _target(
        self, params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key
    ):
        next_prob = self.get_q(target_params, nxtobses, key)
        next_q = self.to_scalar(next_prob)
        if self.munchausen:
            if self.double_q:
                next_action_probs = self.get_q(params, nxtobses, key)
                next_action_q = self.to_scalar(next_action_probs)
                next_sub_q, tau_log_pi_next = q_log_pi(next_action_q, self.munchausen_entropy_tau)
            else:
                next_sub_q, tau_log_pi_next = q_log_pi(next_q, self.munchausen_entropy_tau)
            pi_next = jax.nn.softmax(next_sub_q / self.munchausen_entropy_tau)
            next_vals = (
                jnp.sum(pi_next * (next_q - tau_log_pi_next), axis=1, keepdims=True)
                * not_terminateds
            )

            if self.double_q:
                q_k_targets = self.get_q(params, obses, key)
                q_k_targets = self.to_scalar(q_k_targets)
            else:
                q_k_targets = self.get_q(target_params, obses, key)
                q_k_targets = self.to_scalar(q_k_targets)
            _, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau)
            munchausen_addon = jnp.take_along_axis(tau_log_pi, jnp.squeeze(actions, axis=1), axis=1)

            rewards = rewards + self.munchausen_alpha * jnp.clip(
                munchausen_addon, a_min=-1, a_max=0
            )
        else:
            if self.double_q:
                next_action_probs = self.get_q(params, nxtobses, key)
                next_action_q = self.to_scalar(next_action_probs)
                next_actions = jnp.argmax(next_action_q, axis=1, keepdims=True)
            else:
                next_actions = jnp.argmax(next_q, axis=1, keepdims=True)
            next_vals = not_terminateds * jnp.take_along_axis(next_q, next_actions, axis=1)
        target_q = (next_vals * self._gamma) + rewards
        target_distribution = self.to_probs(target_q)
        return target_distribution

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="HL_GAUSS_C51",
        run_name="HL_GAUSS_C51",
    ):
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            experiment_name,
            run_name,
        )
