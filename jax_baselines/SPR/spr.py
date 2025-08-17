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
    tree_random_normal_like,
)
from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.SPR.efficent_buffer import (
    PrioritizedTransitionReplayBuffer,
    TransitionReplayBuffer,
)


class SPR(Q_Network_Family):
    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        off_policy_fix=False,
        scaled_by_reset=False,
        categorial_bar_n=51,
        categorial_max=250,
        categorial_min=-250,
        **kwargs
    ):

        self.shift_size = 4
        self.prediction_depth = 5
        self.off_policy_fix = off_policy_fix
        self.scaled_by_reset = scaled_by_reset
        self.intensity_scale = 0.05
        self.categorial_bar_n = categorial_bar_n
        self.categorial_max = float(categorial_max)
        self.categorial_min = float(categorial_min)

        # Set SPR-specific defaults
        spr_kwargs = {
            "exploration_fraction": 0,
            "exploration_final_eps": 0,
            "exploration_initial_eps": 0,
            "train_freq": 1,
            "double_q": True,
            "dueling_model": True,
            "n_step": 3,  # n_step 3 is better than 10 in breakout
            "prioritized_replay": True,
            "param_noise": True,
            "target_network_update_freq": 0,
            **kwargs,
        }

        self.name = "SPR"
        super().__init__(env_builder, model_builder_maker, **spr_kwargs)

        self._gamma = jnp.power(self.gamma, jnp.arange(self.n_step))

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
        self.target_params = tree_random_normal_like(next(self.key_seq), self.params)
        if self.scaled_by_reset:
            self.reset_hardsoft = filter_like_tree(
                self.params,
                "qnet",
                (lambda x, filtered: jnp.ones_like(x) if filtered else jnp.ones_like(x) * 0.2),
            )  # hard_reset for qnet and scaled_by_reset for the rest
            self.soft_reset_freq = 40000
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

    def actions(self, obs, epsilon, eval_mode=False):
        params_to_use = self.target_params if self.scaled_by_reset else self.params
        if eval_mode and self.use_checkpointing and self.checkpointing_enabled:
            params_to_use = self.checkpoint_params
        if epsilon <= np.random.uniform(0, 1):
            actions = np.asarray(
                self._get_actions(
                    params_to_use,
                    obs,
                    next(self.key_seq) if self.param_noise else None,
                )
            )
        else:
            actions = np.random.choice(self.action_size[0], [self.worker_size, 1])
        return actions

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
        # SPR has a more complex structure, so we handle it specially
        # Use fixed chunk size based on base gradient_steps to avoid JIT recompilation
        fixed_chunk_size = (
            self.gradient_steps
        )  # Use base gradient_steps for consistent JIT compilation

        # Calculate how many chunks we need
        num_chunks = gradient_steps // fixed_chunk_size

        total_loss = 0.0
        total_rprloss = 0.0
        total_t_mean = 0.0

        # Process all chunks
        for _ in range(num_chunks):
            data = self._sample_batch(fixed_chunk_size * self.batch_size)

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

            self.train_steps_count += fixed_chunk_size
            self._update_priorities(data, new_priorities)

            total_loss += loss
            total_rprloss += rprloss
            total_t_mean += t_mean

        # Average the losses across all chunks
        loss = total_loss / num_chunks
        rprloss = total_rprloss / num_chunks
        t_mean = total_t_mean / num_chunks

        if self.logger_run and (steps - self._last_log_step >= self.log_interval):
            self._last_log_step = steps
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
            lambda o: jnp.argmax(
                jnp.sum(
                    self.get_q(params, o, key) * self._categorial_bar,
                    axis=2,
                ),
                axis=1,
            ),
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
            target_q = jnp.sum(
                target_distribution * self.categorial_bar,
                axis=1,
            )
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
                gammas * not_terminateds, axis=2
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
            target_categorial = gammas * not_terminateds * self.categorial_bar + rewards  # [32, 51]
            target_distribution = tdist(next_distribution, target_categorial)

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
        experiment_name="SPR",
        run_name="SPR",
    ):
        super().learn(total_timesteps, callback, log_interval, experiment_name, run_name)
