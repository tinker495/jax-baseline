import jax
import jax.numpy as jnp
import optax

from jax_baselines.BBF.bbf import BBF
from jax_baselines.common.utils import (
    convert_jax,
    filter_like_tree,
    q_log_pi,
    scaled_by_reset_with_filter,
    soft_update,
    tree_random_normal_like,
)
from jax_baselines.DQN.lifecycle import QNetTrainResult


class HL_GAUSS_BBF(BBF):
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

        self.shift_size = 4
        self.prediction_depth = 5
        self.off_policy_fix = off_policy_fix
        self.intensity_scale = 0.05
        self.sigma = 0.75
        self.spr_weight = float(spr_weight)
        self.categorial_bar_n = categorial_bar_n
        self.categorial_max = float(categorial_max)
        self.categorial_min = float(categorial_min)

        # Set HL_GAUSS_BBF-specific defaults
        hl_gauss_bbf_kwargs = {
            "double_q": True,
            "dueling_model": True,
            "n_step": 10,
            "prioritized_replay": True,
            "target_network_update_freq": 0,
            **kwargs,
        }

        super().__init__(env_builder, model_builder_maker, **hl_gauss_bbf_kwargs)

        self.name = "HL_GAUSS_BBF"

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
        self.optimizer = optax.adamw(learning_rate=self.learning_rate, weight_decay=0.1)
        self.opt_state = self.optimizer.init(self.params)

        self.support = jnp.linspace(
            self.categorial_min,
            self.categorial_max,
            self.categorial_bar_n + 1,
            dtype=jnp.float32,
        )
        bin_width = self.support[1] - self.support[0]
        self.sigma = self.sigma * bin_width

        # Use common JIT compilation
        self._compile_common_functions()

    def _train_on_batch(self, data, context):
        (
            self.params,
            self.target_params,
            self.opt_state,
            loss,
            t_mean,
            new_priorities,
            rprloss,
        ) = self._train_step(
            self.params,
            self.target_params,
            self.opt_state,
            context.train_steps_count,
            next(self.key_seq),
            **data,
        )

        return QNetTrainResult.from_values(
            loss=loss,
            target=t_mean,
            replay_priorities=new_priorities,
            metrics={"loss/rprloss": rprloss},
        )

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
            self.to_scalar(self.get_q(params, convert_jax(obses), key)),
            axis=1,
            keepdims=True,
        )

    def get_scheduled_gamma_nstep(self, steps):
        ratio = (
            jnp.minimum((steps % self.soft_reset_freq) / self.soft_reset_freq, 1 / 4) * 4
        )  # 0 ~ 1 increasing at 1/4
        n_steps = 10 - jnp.round(7 * ratio)  # 10 ~ 3 decreasing at 1/4
        return n_steps, self.gamma

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
            (loss, (centropy, qloss, rprloss)), grad = jax.value_and_grad(self._loss, has_aux=True)(
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
            target_q = self.to_scalar(jnp.expand_dims(target_distribution, 1)).mean()
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

    def _loss(
        self,
        params,
        target_params,
        obses,
        actions,
        filled,
        parsed_obses,
        parsed_actions,
        target_distribution,
        weights,
        key,
    ):
        rprloss = self._represetation_loss(params, target_params, obses, actions, filled, key)
        distribution = jnp.squeeze(
            jnp.take_along_axis(self.get_q(params, parsed_obses, key), parsed_actions, axis=1)
        )
        centropy = -jnp.sum(target_distribution * jnp.log(distribution + 1e-6), axis=1)
        mean_centropy = jnp.mean(centropy)
        total_loss = mean_centropy + rprloss
        return total_loss, (
            centropy,
            mean_centropy,
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
        target_q = (next_vals * gammas) + rewards
        target_distribution = self.to_probs(target_q)
        return target_distribution

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="HL_GAUSS_BBF",
        run_name="HL_GAUSS_BBF",
    ):
        super().learn(total_timesteps, callback, log_interval, experiment_name, run_name)
