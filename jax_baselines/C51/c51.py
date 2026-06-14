from copy import deepcopy

import jax
import jax.numpy as jnp
import optax

from jax_baselines.common.distributional import (
    CategoricalBackend,
    MunchausenSpec,
    distributional_td_target,
)
from jax_baselines.common.jax_utils import convert_jax
from jax_baselines.common.param_updates import hard_update
from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.DQN.training import QNetTrainResult


class C51(Q_Network_Family):
    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        categorial_bar_n=51,
        categorial_max=250,
        categorial_min=-250,
        **kwargs,
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

        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.dueling_model,
            self.param_noise,
            self.categorial_bar_n,
            self.policy_kwargs,
        )

        self.preproc, self.model, self.params = model_builder(next(self.key_seq), print_model=True)
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

    def _train_on_batch(self, data, context):
        (
            self.params,
            self.target_params,
            self.opt_state,
            loss,
            t_mean,
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
            loss=loss, target=t_mean, replay_priorities=new_priorities
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
            params,
            target_params,
            obses,
            actions,
            rewards,
            nxtobses,
            not_terminateds,
            key,
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
        next_dists = self.get_q(target_params, nxtobses, key)
        online_next_dists = self.get_q(params, nxtobses, key) if self.double_q else None

        backend = CategoricalBackend(
            support=self.categorial_bar[0],
            support_min=self.categorial_min,
            support_max=self.categorial_max,
            delta=self.delta_bar,
            n_bins=self.categorial_bar_n,
        )

        if self.munchausen:
            behavior_dists = self.get_q(params if self.double_q else target_params, obses, key)
            munchausen = MunchausenSpec(
                alpha=self.munchausen_alpha, tau=self.munchausen_entropy_tau
            )
        else:
            behavior_dists = None
            munchausen = None

        return distributional_td_target(
            next_dists=next_dists,
            actions=jnp.squeeze(actions, axis=2),
            reward=rewards,
            not_terminated=not_terminateds,
            gamma=self._gamma,
            backend=backend,
            online_next_dists=online_next_dists,
            behavior_dists=behavior_dists,
            munchausen=munchausen,
        )

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="C51",
        run_name="C51",
        eval_num=100,
    ):
        super().learn(total_timesteps, callback, log_interval, experiment_name, run_name, eval_num)
