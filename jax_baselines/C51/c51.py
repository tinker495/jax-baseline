from copy import deepcopy

import jax
import jax.numpy as jnp
import optax

from jax_baselines.common.utils import convert_jax, hard_update, q_log_pi
from jax_baselines.DQN.base_class import Q_Network_Family


class C51(Q_Network_Family):
    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        categorial_bar_n=51,
        categorial_max=250,
        categorial_min=-250,
        **kwargs
    ):
        # Initialize subclass-specific attributes BEFORE calling super().__init__
        # because the base class constructor may call self.setup_model(), which
        # needs these attributes to be present.
        self.name = "C51"
        self.categorial_bar_n = categorial_bar_n
        self.categorial_max = float(categorial_max)
        self.categorial_min = float(categorial_min)

        super().__init__(env_builder, model_builder_maker, **kwargs)

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

        self.categorial_bar = jnp.expand_dims(
            jnp.linspace(self.categorial_min, self.categorial_max, self.categorial_bar_n),
            axis=0,
        )  # [1, 51]
        self._categorial_bar = jnp.expand_dims(self.categorial_bar, axis=0)  # [1, 1, 51]
        self.delta_bar = jax.device_put(
            (self.categorial_max - self.categorial_min) / (self.categorial_bar_n - 1)
        )

        # Use common JIT compilation
        self._compile_common_functions()

    def get_q(self, params, obses, key=None) -> jnp.ndarray:
        return self.model(params, key, self.preproc(params, key, obses))

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        return jnp.expand_dims(
            jnp.argmax(
                jnp.sum(
                    self.get_q(params, convert_jax(obses), key) * self._categorial_bar,
                    axis=2,
                ),
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
            jnp.mean(
                jnp.sum(
                    target_distribution * self.categorial_bar,
                    axis=1,
                )
            ),
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
        next_distributions = self.get_q(target_params, nxtobses, key)
        if self.double_q:
            next_action_q = jnp.sum(
                self.get_q(params, nxtobses, key) * self._categorial_bar, axis=2
            )
        else:
            next_action_q = jnp.sum(next_distributions * self._categorial_bar, axis=2)

        def tdist(next_distribution, target_categorial):
            Tz = jnp.clip(
                target_categorial, self.categorial_min, self.categorial_max
            )  # clip to range of bar
            C51_B = ((Tz - self.categorial_min) / self.delta_bar).astype(
                jnp.float32
            )  # bar index as float
            C51_L = jnp.floor(C51_B).astype(jnp.int32)  # bar lower index as int
            C51_H = jnp.ceil(C51_B).astype(jnp.int32)  # bar higher index as int

            def project_one(p, b, _l, _u):
                exact = _l == _u
                m = jnp.zeros((self.categorial_bar_n,), dtype=p.dtype)

                w_l = jnp.where(exact, p, p * (_u.astype(jnp.float32) - b))
                w_u = jnp.where(exact, jnp.zeros_like(p), p * (b - _l.astype(jnp.float32)))

                m = m.at[_l].add(w_l)
                m = m.at[_u].add(w_u)
                return m

            return jax.vmap(project_one, in_axes=(0, 0, 0, 0))(
                next_distribution, C51_B, C51_L, C51_H
            )

        if self.munchausen:
            next_sub_q, tau_log_pi_next = q_log_pi(next_action_q, self.munchausen_entropy_tau)
            pi_next = jax.nn.softmax(next_sub_q / self.munchausen_entropy_tau)  # [32, action_size]
            next_categorials = self._categorial_bar - jnp.expand_dims(
                tau_log_pi_next, axis=2
            )  # [32, action_size, 51]

            if self.double_q:
                q_k_targets = jnp.sum(self.get_q(params, obses, key) * self.categorial_bar, axis=2)
            else:
                q_k_targets = jnp.sum(
                    self.get_q(target_params, obses, key) * self.categorial_bar, axis=2
                )
            _, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau)
            munchausen_addon = jnp.take_along_axis(tau_log_pi, jnp.squeeze(actions, axis=2), axis=1)

            rewards = rewards + self.munchausen_alpha * jnp.clip(
                munchausen_addon, a_min=-1, a_max=0
            )  # [32, 1]
            target_categorials = jnp.expand_dims(
                self._gamma * not_terminateds, axis=2
            ) * next_categorials + jnp.expand_dims(
                rewards, axis=2
            )  # [32, action_size, 51]
            target_distributions = jax.vmap(tdist, in_axes=(1, 1), out_axes=1)(
                next_distributions, target_categorials
            )
            target_distribution = jnp.sum(
                jnp.expand_dims(pi_next, axis=2) * target_distributions, axis=1
            )
        else:
            next_actions = jnp.expand_dims(jnp.argmax(next_action_q, axis=1), axis=(1, 2))
            next_distribution = jnp.squeeze(
                jnp.take_along_axis(next_distributions, next_actions, axis=1)
            )
            target_categorial = (
                self._gamma * not_terminateds * self.categorial_bar + rewards
            )  # [32, 51]
            target_distribution = tdist(next_distribution, target_categorial)

        return target_distribution

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="C51",
        run_name="C51",
    ):
        super().learn(total_timesteps, callback, log_interval, experiment_name, run_name)
