from copy import deepcopy

import jax
import jax.numpy as jnp
import optax

from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.math.distributional import (
    HLGaussBackend,
    HLGaussTransform,
    MunchausenSpec,
    distributional_td_target,
)
from jax_baselines.math.jax_utils import convert_jax
from jax_baselines.math.param_updates import hard_update


class HL_GAUSS_C51(Q_Network_Family):
    _run_name = "HL_GAUSS_C51"
    supports_bulk_training = True

    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        categorial_bar_n=51,
        categorial_max=250,
        categorial_min=-250,
        **kwargs,
    ):

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

        self.hl_gauss = HLGaussTransform.build(
            self.categorial_min, self.categorial_max, self.categorial_bar_n
        )

        # Use common JIT compilation
        self._compile_common_functions()
        self._bulk_scan = jax.jit(self._bulk_scan)

    def get_q(self, params, obses, key=None) -> jnp.ndarray:
        return self.model(params, key, self.preproc(params, key, obses))

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        return jnp.argmax(
            self.hl_gauss.to_scalar(self.get_q(params, convert_jax(obses), key)),
            axis=1,
            keepdims=True,
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
            self.hl_gauss.to_scalar(jnp.expand_dims(target_distribution, 1)).mean(),
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
        munchausen = (
            MunchausenSpec(alpha=self.munchausen_alpha, tau=self.munchausen_entropy_tau)
            if self.munchausen
            else None
        )
        behavior_dists = (
            (
                self.get_q(params, obses, key)
                if self.double_q
                else self.get_q(target_params, obses, key)
            )
            if self.munchausen
            else None
        )
        return distributional_td_target(
            next_dists=next_dists,
            actions=jnp.squeeze(actions, axis=1),
            reward=rewards,
            not_terminated=not_terminateds,
            gamma=self._gamma,
            backend=HLGaussBackend(self.hl_gauss),
            online_next_dists=online_next_dists,
            behavior_dists=behavior_dists,
            munchausen=munchausen,
        )
