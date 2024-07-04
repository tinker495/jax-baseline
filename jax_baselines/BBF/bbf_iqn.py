import dm_pix as pix
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.common.losses import QuantileHuberLosses
from jax_baselines.common.utils import (
    convert_jax,
    filter_like_tree,
    q_log_pi,
    soft_reset,
    soft_update,
    tree_random_normal_like,
)
from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.SPR.efficent_buffer import (
    PrioritizedTransitionReplayBuffer,
    TransitionReplayBuffer,
)


class BBF_IQN(Q_Network_Family):
    def __init__(
        self,
        env,
        model_builder_maker,
        gamma=0.995,
        learning_rate=3e-4,
        buffer_size=100000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        exploration_initial_eps=1.0,
        n_support=32,
        delta=0.1,
        train_freq=1,
        gradient_steps=1,
        batch_size=32,
        off_policy_fix=False,
        learning_starts=1000,
        CVaR=1.0,
        param_noise=False,
        munchausen=False,
        log_interval=200,
        tensorboard_log=None,
        _init_setup_model=True,
        policy_kwargs=None,
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
        compress_memory=False,
    ):

        self.name = "SPR"
        self.shift_size = 4
        self.prediction_depth = 5
        self.off_policy_fix = off_policy_fix
        self.intensity_scale = 0.05
        self.CVaR = CVaR
        self.risk_avoid = CVaR != 1.0
        self.n_support = n_support
        self.delta = delta

        super().__init__(
            env,
            model_builder_maker,
            gamma,
            learning_rate,
            buffer_size,
            exploration_fraction,
            exploration_final_eps,
            exploration_initial_eps,
            train_freq,
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
            param_noise,
            munchausen,
            log_interval,
            tensorboard_log,
            _init_setup_model,
            policy_kwargs,
            full_tensorboard_log,
            seed,
            optimizer,
            compress_memory,
        )

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
            (lambda x, filtered: jnp.ones_like(x) if filtered else jnp.ones_like(x) * 0.5),
        )  # hard_reset for qnet and soft_reset for the rest
        self.soft_reset_freq = 40000
        self.optimizer = optax.adamw(learning_rate=self.learning_rate, weight_decay=0.1)
        self.opt_state = self.optimizer.init(self.params)

        self.get_q = jax.jit(self.get_q)
        self._get_actions = jax.jit(self._get_actions)
        self._loss = jax.jit(self._loss)
        self._target = jax.jit(self._target)
        self._train_step = jax.jit(self._train_step)

    def actions(self, obs, epsilon):
        if epsilon <= np.random.uniform(0, 1):
            actions = np.asarray(
                self._get_actions(
                    self.target_params,
                    obs,
                    next(self.key_seq),
                )
            )
        else:
            actions = np.random.choice(self.action_size[0], [self.worker_size, 1])
        return actions

    def get_q(self, params, obses, tau, key=None) -> jnp.ndarray:
        return self.model(params, key, self.preproc(params, key, obses), tau)

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        tau = jax.random.uniform(key, (self.worker_size, self.n_support)) * self.CVaR
        return jnp.expand_dims(
            jnp.argmax(
                jnp.sum(
                    self.get_q(params, convert_jax(obses), tau, key),
                    axis=2,
                ),
                axis=1,
            ),
            axis=1,
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

        if self.summary and steps % self.log_interval == 0:
            self.summary.add_scalar("loss/qloss", loss, steps)
            self.summary.add_scalar("loss/rprloss", rprloss, steps)
            self.summary.add_scalar("loss/targets", t_mean, steps)

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
        return n_steps, gamma  # self.gamma

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
            key1, key2, subkey = jax.random.split(key, 3)
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
                key1,
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
                key2,
            )
            updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            target_params = soft_update(params, target_params, 0.005)
            params = soft_reset(params, key1, steps, self.soft_reset_freq, self.reset_hardsoft)
            target_q = jnp.mean(target_distribution)
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
        tau = jax.random.uniform(key, (self.batch_size, self.n_support))
        theta_loss_tile = jnp.take_along_axis(
            self.get_q(params, parsed_obses, tau, key), parsed_actions, axis=1
        )  # batch x 1 x support
        logit_valid_tile = jnp.expand_dims(target_distribution, axis=2)  # batch x support x 1
        batched_quantileloss = QuantileHuberLosses(
            logit_valid_tile, theta_loss_tile, jnp.expand_dims(tau, axis=1), self.delta
        )
        mean_quantileloss = jnp.mean(batched_quantileloss)
        total_loss = mean_quantileloss + rprloss
        return total_loss, (
            batched_quantileloss,
            mean_quantileloss,
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
        target_tau = jax.random.uniform(key, (self.batch_size, self.n_support))
        next_q = self.get_q(target_params, nxtobses, target_tau, key)

        if self.munchausen:
            if self.double_q:
                next_q_mean = jnp.mean(self.get_q(params, nxtobses, target_tau, key), axis=2)
            else:
                next_q_mean = jnp.mean(next_q, axis=2)
            next_sub_q, tau_log_pi_next = q_log_pi(next_q_mean, self.munchausen_entropy_tau)
            pi_next = jnp.expand_dims(
                jax.nn.softmax(next_sub_q / self.munchausen_entropy_tau), axis=2
            )  # batch x actions x 1
            next_vals = next_q - jnp.expand_dims(
                tau_log_pi_next, axis=2
            )  # batch x actions x support
            sample_pi = jax.random.categorical(
                key, jnp.tile(pi_next, (1, 1, self.n_support)), 1
            )  # batch x 1 x support
            next_vals = jnp.take_along_axis(
                next_vals, jnp.expand_dims(sample_pi, axis=1), axis=1
            )  # batch x 1 x support
            next_vals = not_terminateds * jnp.squeeze(next_vals, axis=1)

            if self.double_q:
                q_k_targets = jnp.mean(self.get_q(params, obses, target_tau, key), axis=2)
            else:
                q_k_targets = jnp.mean(self.get_q(target_params, obses, target_tau, key), axis=2)
            _, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau)
            munchausen_addon = jnp.take_along_axis(tau_log_pi, jnp.squeeze(actions, axis=2), axis=1)

            rewards = rewards + self.munchausen_alpha * jnp.clip(
                munchausen_addon, a_min=-1, a_max=0
            )
        else:
            if self.double_q:
                next_actions = jnp.argmax(
                    jnp.mean(self.get_q(params, nxtobses, target_tau, key), axis=2, keepdims=True),
                    axis=1,
                    keepdims=True,
                )
            else:
                next_actions = jnp.argmax(
                    jnp.mean(next_q, axis=2, keepdims=True), axis=1, keepdims=True
                )
            next_vals = not_terminateds * jnp.squeeze(
                jnp.take_along_axis(next_q, next_actions, axis=1)
            )  # batch x support
        return (next_vals * gammas) + rewards  # batch x support

    def tb_log_name_update(self, tb_log_name):
        tb_log_name = tb_log_name + (
            "({:d})_CVaR({:.2f})".format(self.n_support, self.CVaR)
            if self.risk_avoid
            else "({:d})".format(self.n_support)
        )
        if self.munchausen:
            tb_log_name = "M-" + tb_log_name
        if self.param_noise:
            tb_log_name = "Noisy_" + tb_log_name
        return tb_log_name

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=100,
        tb_log_name="BBF_IQN",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            tb_log_name,
            reset_num_timesteps,
            replay_wrapper,
        )
