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
        double_q=True,
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
        categorial_bar_n=51,
        categorial_max=250,
        categorial_min=-250,
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

        self.name = "C51"
        self.categorial_bar_n = categorial_bar_n
        self.categorial_max = float(categorial_max)
        self.categorial_min = float(categorial_min)

        if _init_setup_model:
            self.setup_model()

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

            # Handle the case where Tz exactly matches an atom
            # If C51_L == C51_H, it means Tz exactly matches an atom
            exact_match = C51_L == C51_H

            def tdist(next_distribution, C51_L, C51_H, C51_b, exact_match):
                target_distribution = jnp.zeros((self.categorial_bar_n))

                # If exact match, assign all probability to that atom
                target_distribution = jnp.where(
                    exact_match,
                    target_distribution.at[C51_L].add(next_distribution),
                    target_distribution,
                )

                # If not exact match, use linear interpolation between two closest atoms
                target_distribution = jnp.where(
                    ~exact_match,
                    target_distribution.at[C51_L].add(
                        next_distribution * (C51_H.astype(jnp.float32) - C51_b)
                    ),
                    target_distribution,
                )
                target_distribution = jnp.where(
                    ~exact_match,
                    target_distribution.at[C51_H].add(
                        next_distribution * (C51_b - C51_L.astype(jnp.float32))
                    ),
                    target_distribution,
                )

                return target_distribution

            return jax.vmap(tdist, in_axes=(0, 0, 0, 0, 0))(
                next_distribution, C51_L, C51_H, C51_B, exact_match
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
