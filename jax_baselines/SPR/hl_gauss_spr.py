from copy import deepcopy

import dm_pix as pix
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.common.utils import (
    convert_jax,
    filter_like_tree,
    q_log_pi,
    scaled_by_reset_with_filter,
    soft_update,
)
from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.SPR.efficent_buffer import (
    PrioritizedTransitionReplayBuffer,
    TransitionReplayBuffer,
)


class HL_GAUSS_SPR(Q_Network_Family):
    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        num_workers=1,
        eval_eps=20,
        gamma=0.995,
        learning_rate=3e-4,
        buffer_size=100000,
        gradient_steps=1,
        batch_size=32,
        off_policy_fix=False,
        scaled_by_reset=False,
        learning_starts=1000,
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

        self.shift_size = 4
        self.prediction_depth = 5
        self.off_policy_fix = off_policy_fix
        self.scaled_by_reset = scaled_by_reset
        self.intensity_scale = 0.05
        self.sigma = 0.75
        self.categorial_bar_n = categorial_bar_n
        self.categorial_max = float(categorial_max)
        self.categorial_min = float(categorial_min)

        super().__init__(
            env_builder,
            model_builder_maker,
            num_workers,
            eval_eps,
            gamma,
            learning_rate,
            buffer_size,
            0,
            0,
            0,
            1,
            gradient_steps,
            batch_size,
            True,
            True,
            10,
            learning_starts,
            0,
            True,
            0.6,
            0.4,
            1e-3,
            True,
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

        self.name = "HL_GAUSS_SPR"

        self._gamma = jnp.power(self.gamma, jnp.arange(self.n_step))

        if _init_setup_model:
            self.setup_model()

    def get_memory_setup(self):
        if self.prioritized_replay:
            self.replay_buffer = PrioritizedTransitionReplayBuffer(
                self.buffer_size,
                self.observation_space,
                1,
                prediction_depth=max(self.prediction_depth, self.n_step),
                alpha=self.prioritized_replay_alpha,
                eps=self.prioritized_replay_eps,
            )
        else:
            self.replay_buffer = TransitionReplayBuffer(
                self.buffer_size,
                self.observation_space,
                1,
                prediction_depth=max(self.prediction_depth, self.n_step),
            )

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
        self.target_params = deepcopy(self.params)
        if self.scaled_by_reset:
            self.reset_hardsoft = filter_like_tree(
                self.params,
                "qnet",
                (lambda x, filtered: jnp.ones_like(x) if filtered else jnp.ones_like(x) * 0.2),
            )  # hard_reset for qnet and scaled_by_reset for the rest
            self.soft_reset_freq = 40000

        self.opt_state = self.optimizer.init(self.params)

        self.support = jnp.linspace(
            self.categorial_min, self.categorial_max, self.categorial_bar_n + 1, dtype=jnp.float32
        )
        bin_width = self.support[1] - self.support[0]
        self.sigma = self.sigma * bin_width

        self.get_q = jax.jit(self.get_q)
        self._get_actions = jax.jit(self._get_actions)
        self._loss = jax.jit(self._loss)
        self._target = jax.jit(self._target)
        self._train_step = jax.jit(self._train_step)

    def actions(self, obs, epsilon):
        if epsilon <= np.random.uniform(0, 1):
            actions = np.asarray(
                self._get_actions(
                    self.target_params if self.scaled_by_reset else self.params,
                    obs,
                    next(self.key_seq) if self.param_noise else None,
                )
            )
        else:
            actions = np.random.choice(self.action_size[0], [self.worker_size, 1])
        return actions

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
        # Sample a batch from the replay buffer

        if self.prioritized_replay:
            data = self.replay_buffer.sample(
                gradient_steps * self.batch_size, self.prioritized_replay_beta0
            )
        else:
            data = self.replay_buffer.sample(gradient_steps * self.batch_size)

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
            self.train_steps_count,
            next(self.key_seq),
            **data,
        )

        self.train_steps_count += gradient_steps

        if self.prioritized_replay:
            self.replay_buffer.update_priorities(data["indexes"], new_priorities)

        if self.logger_run and steps % self.log_interval == 0:
            self.logger_run.log_metric("loss/qloss", loss, steps)
            self.logger_run.log_metric("loss/rprloss", rprloss, steps)
            self.logger_run.log_metric("loss/targets", t_mean, steps)

        return loss

    def _image_augmentation(self, obs, key):
        """Random augmentation for input images.

        Args:
            obses (np.ndarray): input images  B x K x H x W x C
            key (jax.random.PRNGKey): random key

        Returns:
            list(np.ndarray): augmented images
        """

        def random_shift(obs, key):  # K x H x W x C
            obs = jnp.pad(
                obs,
                ((self.shift_size, self.shift_size), (self.shift_size, self.shift_size), (0, 0)),
                mode="constant",
            )
            obs = pix.random_crop(
                key,
                obs,
                (
                    obs.shape[0] - self.shift_size * 2,
                    obs.shape[1] - self.shift_size * 2,
                    obs.shape[2],
                ),
            )
            return obs

        def Intensity(obs, key):
            noise = (
                1.0 + jnp.clip(jax.random.normal(key, (1, 1, 1)), -2.0, 2.0) * self.intensity_scale
            )
            return obs * noise

        def augment(obs, key):
            subkey1, subkey2 = jax.random.split(key)
            obs_len = obs.shape[0]
            obs = jax.vmap(random_shift)(obs, jax.random.split(subkey1, obs_len))
            obs = jax.vmap(Intensity)(obs, jax.random.split(subkey2, obs_len))
            return obs

        batch_size = obs.shape[0]
        return jax.vmap(augment)(obs, jax.random.split(key, batch_size))

    def get_last_idx(self, params, obses, actions, filled, key):
        if self.n_step == 1:
            return jnp.zeros((obses[0].shape[0], 1), jnp.int32), jnp.ones(
                (obses[0].shape[0], 1), jnp.bool_
            )
        parsed_filled = jnp.reshape(filled[:, : self.n_step], (-1, self.n_step))
        last_idxs = jnp.argmax(
            parsed_filled * jnp.arange(1, self.n_step + 1), axis=1, keepdims=True
        )
        if not self.off_policy_fix:
            return last_idxs, parsed_filled
        action_obs = [o[:, 1 : self.n_step] for o in obses]  # 1 ~ n_step
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

        def f(updates, input):
            params, target_params, opt_state, key = updates
            obses, actions, rewards, not_terminateds, filled, weights, steps = input
            key, subkey = jax.random.split(key)
            parsed_obses = [jnp.reshape(o[:, 0], (-1, *o.shape[2:])) for o in obses]
            last_idxs, parsed_filled = self.get_last_idx(target_params, obses, actions, filled, key)
            parsed_nxtobses = [
                jnp.reshape(
                    jnp.take_along_axis(o, jnp.reshape(last_idxs + 1, (-1, 1, 1, 1, 1)), axis=1),
                    (-1, *o.shape[2:]),
                )
                for o in obses
            ]
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
            target_q = self.to_scalar(jnp.expand_dims(target_distribution, 1)).mean()
            return (params, target_params, opt_state, subkey), (centropy, qloss, rprloss, target_q)

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
        )  # jnp.sum(jnp.abs(error) * filled, axis=-1) / jnp.sum(filled, axis=-1)

    def _represetation_loss(self, params, target_params, obses, actions, filled, key):
        initial_obs = [o[:, 0] for o in obses]  # B x H x W x C
        target_obs = [o[:, 1:] for o in obses]  # B x K x H x W x C
        initial_features = self.preproc(params, key, initial_obs)  # B x D
        target_features = jax.vmap(self.preproc, in_axes=(None, None, 1), out_axes=1)(
            target_params, key, target_obs
        )  # B x K x D
        traget_projection = jax.vmap(self.projection, in_axes=(None, None, 1), out_axes=1)(
            target_params, key, target_features
        )  # B x K x D
        traget_projection = traget_projection / jnp.linalg.norm(
            traget_projection, axis=-1, keepdims=True
        )  # normalize

        def body(carry, x):
            loss, current_features = carry
            action, filled, target_projection = x
            action = jax.nn.one_hot(jnp.squeeze(action), self.action_size[0])
            current_features = self.transition(params, key, current_features, action)
            current_projection = self.projection(params, key, current_features)
            current_projection = self.prediction(params, key, current_projection)
            current_projection = current_projection / jnp.linalg.norm(
                current_projection, axis=-1, keepdims=True
            )  # normalize
            cosine_similarity = jnp.sum(current_projection * target_projection, axis=-1) - 1.0
            loss = loss - cosine_similarity * filled
            return (loss, current_features), None

        actions = jnp.swapaxes(actions, 0, 1)
        filled = jnp.swapaxes(filled, 0, 1)
        traget_projection = jnp.swapaxes(traget_projection, 0, 1)
        (loss, _), _ = jax.lax.scan(
            body,
            (jnp.zeros((initial_features.shape[0],)), initial_features),
            (actions, filled, traget_projection),
        )
        return jnp.mean(loss)

    def _target(
        self, params, target_params, obses, actions, rewards, nxtobses, not_terminateds, gammas, key
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

    def run_name_update(self, run_name):
        if self.scaled_by_reset:
            run_name = "SR-" + run_name
        if self.munchausen:
            run_name = "M-" + run_name
        if self.off_policy_fix:
            n_step_str = "OF_"
            run_name = n_step_str + run_name
        return run_name

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="HL_GAUSS_SPR",
        run_name="HL_GAUSS_SPR",
    ):
        super().learn(total_timesteps, callback, log_interval, experiment_name, run_name)
