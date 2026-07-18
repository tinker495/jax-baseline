from copy import deepcopy

import jax
import jax.numpy as jnp
import optax

from jax_baselines.math.distributional import (
    HLGaussBackend,
    HLGaussTransform,
    MunchausenSpec,
    distributional_td_target,
)
from jax_baselines.math.jax_utils import convert_jax
from jax_baselines.math.param_updates import (
    filter_like_tree,
    scaled_by_reset_with_filter,
    soft_update,
)
from jax_baselines.SPR.spr import SPR


class HL_GAUSS_SPR(SPR):
    _run_name = "HL_GAUSS_SPR"

    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        off_policy_fix=False,
        spr_weight=5.0,
        scaled_by_reset=False,
        categorial_bar_n=51,
        categorial_max=250,
        categorial_min=-250,
        **kwargs,
    ):
        # Set HL_GAUSS_SPR-specific defaults. off_policy_fix/spr_weight/scaled_by_reset/
        # categorial_* (and shift_size/prediction_depth/intensity_scale) are owned by
        # SPR.__init__; setting them on self here would be clobbered when super()
        # reapplies SPR's defaults.
        hl_gauss_spr_kwargs = {
            "off_policy_fix": off_policy_fix,
            "spr_weight": spr_weight,
            "scaled_by_reset": scaled_by_reset,
            "categorial_bar_n": categorial_bar_n,
            "categorial_max": categorial_max,
            "categorial_min": categorial_min,
            "exploration_fraction": 0,
            "exploration_final_eps": 0,
            "exploration_initial_eps": 0,
            "train_freq": 1,
            "double_q": True,
            "dueling_model": True,
            # NOTE: diverges from the SPR parent default (n_step=3, tuned "better than
            # 10 in breakout"). Kept at 10 as the HL-Gauss variant's own default; not yet
            # re-tuned. Align to 3 only with experimental confirmation.
            "n_step": 10,
            "prioritized_replay": True,
            "param_noise": True,
            "target_network_update_freq": 0,
            **kwargs,
        }

        super().__init__(env_builder, model_builder_maker, **hl_gauss_spr_kwargs)

        self._gamma = jnp.power(self.gamma, jnp.arange(self.n_step))

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.dueling_model,
            self.param_noise,
            self.categorial_bar_n,
            self.policy_kwargs,
        )
        (
            self.preproc,
            self.model,
            self.transition,
            self.projection,
            self.prediction,
            self.params,
        ) = model_builder(next(self.key_seq), print_model=True)
        # NOTE: diverges from the SPR parent, which seeds the target net with
        # tree_random_normal_like (random re-init). This variant uses a plain copy of
        # the online params. Both are valid target-init regimes; difference is intentional.
        self.target_params = deepcopy(self.params)
        if self.scaled_by_reset:
            self.reset_hardsoft = filter_like_tree(
                self.params,
                "qnet",
                (lambda x, filtered: (jnp.ones_like(x) if filtered else jnp.ones_like(x) * 0.2)),
            )  # hard_reset for qnet and scaled_by_reset for the rest
            self.soft_reset_freq = 40000

        self.opt_state = self.optimizer.init(self.params)

        self.hl_gauss = HLGaussTransform.build(
            self.categorial_min, self.categorial_max, self.categorial_bar_n
        )

        # Use common JIT compilation
        self._compile_common_functions()

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        return jnp.argmax(
            self.hl_gauss.to_scalar(self.get_q(params, convert_jax(obses), key)),
            axis=1,
            keepdims=True,
        )

    def get_last_idx(self, params, obses, actions, filled, key):
        if self.n_step == 1:
            batch_size = next(iter(obses.values())).shape[0]
            return jnp.zeros((batch_size, 1), jnp.int32), jnp.ones((batch_size, 1), jnp.bool_)
        parsed_filled = jnp.reshape(filled[:, : self.n_step], (-1, self.n_step))
        last_idxs = jnp.argmax(
            parsed_filled * jnp.arange(1, self.n_step + 1), axis=1, keepdims=True
        )
        if not self.off_policy_fix:
            return last_idxs, parsed_filled
        action_obs = jax.tree.map(lambda value: value[:, 1 : self.n_step], obses)
        pred_actions = jax.vmap(
            lambda o: self._get_actions(params, o, key).squeeze(),
            in_axes=1,
            out_axes=1,
        )(
            action_obs
        )  # B x n_step - 1
        action_equal = jnp.not_equal(
            pred_actions, jnp.squeeze(actions[:, 1 : self.n_step + 1])
        )  # B x n_step - 1
        minimum_not_equal = 1 + jnp.argmax(
            action_equal * jnp.arange(self.n_step, 1, -1), axis=1, keepdims=True
        )
        last_idxs = jnp.minimum(last_idxs, minimum_not_equal)
        # fill partial filled
        parsed_filled = jnp.less_equal(jnp.arange(0, self.n_step), last_idxs)
        return last_idxs, parsed_filled

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
        terminateds,
        filled,
        weights=1,
        indexes=None,
    ):
        obses = convert_jax(obses)
        actions = actions.astype(jnp.int32)
        not_terminateds = 1.0 - terminateds
        obses = jax.tree.map(
            lambda value: jax.lax.cond(
                len(value.shape) >= 5,
                lambda: self._image_augmentation(value, key),
                lambda: value,
            ),
            obses,
        )

        batch_idxes = jnp.arange(next(iter(obses.values())).shape[0]).reshape(
            -1, self.batch_size
        )  # nbatches x batch_size
        batched_obses = jax.tree.map(lambda value: value[batch_idxes], obses)
        batched_actions = actions[batch_idxes]
        batched_rewards = rewards[batch_idxes]
        batched_not_terminateds = not_terminateds[batch_idxes]
        batched_filled = filled[batch_idxes]
        batched_weights = weights[batch_idxes] if self.prioritized_replay else 1
        gradient_steps = batch_idxes.shape[0]
        batched_steps = steps + jnp.arange(gradient_steps)

        def f(updates, input):
            params, target_params, opt_state, key = updates
            obses, actions, rewards, not_terminateds, filled, weights, steps = input
            key, subkey = jax.random.split(key)
            parsed_obses = jax.tree.map(
                lambda value: jnp.reshape(value[:, 0], (-1, *value.shape[2:])), obses
            )
            last_idxs, parsed_filled = self.get_last_idx(target_params, obses, actions, filled, key)
            parsed_nxtobses = jax.tree.map(
                lambda value: jnp.reshape(
                    jnp.take_along_axis(
                        value,
                        jnp.reshape(last_idxs + 1, (-1, 1, 1, 1, 1)),
                        axis=1,
                    ),
                    (-1, *value.shape[2:]),
                ),
                obses,
            )
            parsed_actions = jnp.reshape(actions[:, 0], (-1, 1, 1))
            rewards = jnp.reshape(rewards[:, : self.n_step], (-1, self.n_step))
            parsed_rewards = jnp.sum(rewards * self._gamma * parsed_filled, axis=1, keepdims=True)
            parsed_not_terminateds = jnp.take_along_axis(not_terminateds, last_idxs, axis=1)
            parsed_gamma = (
                jnp.take_along_axis(jnp.expand_dims(self._gamma, 0), last_idxs, axis=1) * self.gamma
            )
            target_distribution = self._target(
                params,
                target_params,
                parsed_obses,
                parsed_actions,
                parsed_rewards,
                parsed_nxtobses,
                parsed_not_terminateds,
                parsed_gamma,
                key,
            )
            transition_obses = jax.tree.map(
                lambda value: value[:, : (self.prediction_depth + 1)], obses
            )
            transition_actions = actions[:, : self.prediction_depth]
            transition_filled = filled[:, : self.prediction_depth]
            (_, (centropy, qloss, rprloss)), grad = jax.value_and_grad(self._loss, has_aux=True)(
                params,
                target_params,
                transition_obses,
                transition_actions,
                transition_filled,
                parsed_obses,
                parsed_actions,
                target_distribution,
                weights,
                key,
            )
            updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            target_params = soft_update(params, target_params, 0.005)
            if self.scaled_by_reset:
                params, opt_state = scaled_by_reset_with_filter(
                    params,
                    opt_state,
                    self.optimizer,
                    key,
                    steps,
                    self.soft_reset_freq,
                    self.reset_hardsoft,
                )
            target_q = self.hl_gauss.to_scalar(jnp.expand_dims(target_distribution, 1)).mean()
            return (params, target_params, opt_state, subkey), (
                centropy,
                qloss,
                rprloss,
                target_q,
            )

        (params, target_params, opt_state, _), outputs = jax.lax.scan(
            f,
            (params, target_params, opt_state, key),
            (
                batched_obses,
                batched_actions,
                batched_rewards,
                batched_not_terminateds,
                batched_filled,
                batched_weights,
                batched_steps,
            ),
        )
        centropy, qloss, rprloss, target_q = outputs
        qloss = jnp.mean(qloss)
        rprloss = jnp.mean(rprloss)
        target_q = jnp.mean(target_q)
        new_priorities = None
        if self.prioritized_replay:
            # nbatches x batch_size -> flatten
            new_priorities = jnp.hstack(centropy)
        return (
            params,
            target_params,
            opt_state,
            qloss,
            target_q,
            new_priorities,
            rprloss,
        )

    def _target(
        self,
        params,
        target_params,
        obses,
        actions,
        rewards,
        nxtobses,
        not_terminateds,
        gammas,
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
            gamma=gammas,
            backend=HLGaussBackend(self.hl_gauss),
            online_next_dists=online_next_dists,
            behavior_dists=behavior_dists,
            munchausen=munchausen,
        )
