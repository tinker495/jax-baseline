import jax
import jax.numpy as jnp
import optax

from jax_baselines.math.jax_utils import convert_jax
from jax_baselines.math.param_updates import (
    filter_like_tree,
    scaled_by_reset_with_filter,
    soft_update,
    tree_random_normal_like,
)
from jax_baselines.SPR.spr import SPR


class BBF(SPR):
    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        off_policy_fix=False,
        spr_weight=5.0,
        categorial_bar_n=51,
        categorial_max=250,
        categorial_min=-250,
        **kwargs,
    ):
        # Set BBF-specific defaults. off_policy_fix/spr_weight/categorial_* (and
        # shift_size/prediction_depth/intensity_scale) are owned by SPR.__init__;
        # setting them on self here would be clobbered when super() reapplies SPR's
        # defaults.
        bbf_kwargs = {
            "off_policy_fix": off_policy_fix,
            "spr_weight": spr_weight,
            "categorial_bar_n": categorial_bar_n,
            "categorial_max": categorial_max,
            "categorial_min": categorial_min,
            "double_q": True,
            "dueling_model": True,
            "n_step": 10,
            "prioritized_replay": True,
            "target_network_update_freq": 0,
            **kwargs,
        }

        super().__init__(env_builder, model_builder_maker, **bbf_kwargs)

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
        self.target_params = tree_random_normal_like(next(self.key_seq), self.params)
        self.reset_hardsoft = filter_like_tree(
            self.params,
            "qnet",
            (lambda x, filtered: (jnp.ones_like(x) if filtered else jnp.ones_like(x) * 0.5)),
        )  # hard_reset for qnet and scaled_by_reset for the rest
        self.soft_reset_freq = 40000
        self.optimizer = self._make_optimizer(self.learning_rate)
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

    def get_behavior_params(self):
        """BBF uses target_params for behavior (training-time actions)."""
        return self.target_params

    def get_last_idx(self, filled, n_step):
        parsed_filled = jnp.where(jnp.arange(self.n_step) < n_step, filled, 0)
        last_idxs = jnp.argmax(
            parsed_filled * jnp.arange(1, self.n_step + 1), axis=1, keepdims=True
        )
        return last_idxs, parsed_filled

    def get_scheduled_gamma_nstep(self, steps):
        ratio = (
            jnp.minimum((steps % self.soft_reset_freq) / self.soft_reset_freq, 1 / 4) * 4
        )  # 0 ~ 1 increasing at 1/4
        n_steps = 10 - jnp.round(7 * ratio)  # 10 ~ 3 decreasing at 1/4
        start_horizon = 1 / (1 - 0.97)
        end_horizon = 1 / (1 - 0.997)
        horizon = (
            start_horizon + (end_horizon - start_horizon) * ratio
        )  # 0.97 ~ 0.997 increasing at 1/4
        gamma = 1 - 1 / horizon
        return n_steps, gamma

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
        obses = [
            jax.lax.cond(
                len(o.shape) >= 5,
                lambda: self._image_augmentation(o, key),
                lambda: o,
            )
            for o in obses
        ]

        batch_idxes = jnp.arange(obses[0].shape[0]).reshape(
            -1, self.batch_size
        )  # nbatches x batch_size
        batched_obses = [o[batch_idxes] for o in obses]
        batched_actions = actions[batch_idxes]
        batched_rewards = rewards[batch_idxes]
        batched_not_terminateds = not_terminateds[batch_idxes]
        batched_filled = filled[batch_idxes]
        batched_weights = weights[batch_idxes] if self.prioritized_replay else 1
        gradient_steps = batch_idxes.shape[0]
        batched_steps = steps + jnp.arange(gradient_steps)
        n_step, gamma = self.get_scheduled_gamma_nstep(steps)
        _gamma = jnp.power(gamma, jnp.arange(self.n_step))

        def f(updates, input):
            params, target_params, opt_state, key = updates
            obses, actions, rewards, not_terminateds, filled, weights, steps = input
            key, subkey = jax.random.split(key)
            parsed_obses = [jnp.reshape(o[:, 0], (-1, *o.shape[2:])) for o in obses]
            last_idxs, parsed_filled = self.get_last_idx(filled, n_step)
            parsed_nxtobses = [
                jnp.reshape(
                    jnp.take_along_axis(o, jnp.reshape(last_idxs + 1, (-1, 1, 1, 1, 1)), axis=1),
                    (-1, *o.shape[2:]),
                )
                for o in obses
            ]
            parsed_actions = jnp.reshape(actions[:, 0], (-1, 1, 1))
            parsed_rewards = jnp.sum(rewards * _gamma * parsed_filled, axis=1, keepdims=True)
            parsed_not_terminateds = jnp.take_along_axis(not_terminateds, last_idxs, axis=1)
            parsed_gamma = (
                jnp.take_along_axis(jnp.expand_dims(_gamma, 0), last_idxs, axis=1) * gamma
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
            transition_obses = [o[:, : (self.prediction_depth + 1)] for o in obses]
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
            params, opt_state = scaled_by_reset_with_filter(
                params,
                opt_state,
                self.optimizer,
                key,
                steps,
                self.soft_reset_freq,
                self.reset_hardsoft,
            )
            target_q = jnp.sum(
                target_distribution * self.categorial_bar,
                axis=1,
            )
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

    def run_name_update(self, run_name):
        if self.munchausen:
            run_name = "M-" + run_name
        if self.param_noise:
            run_name = "Noisy_" + run_name
        return run_name

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="BBF",
        run_name="BBF",
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
