from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.common.losses import hubberloss
from jax_baselines.common.utils import convert_jax, hard_update, scaled_by_reset
from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family


class TD7(Deteministic_Policy_Gradient_Family):
    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        num_workers=1,
        eval_eps=20,
        gamma=0.995,
        learning_rate=3e-4,
        buffer_size=100000,
        target_action_noise_mul=2.0,
        action_noise=0.1,
        train_freq=1,
        gradient_steps=1,
        batch_size=32,
        policy_delay=2,
        learning_starts=1000,
        target_network_update_freq=250,
        prioritized_replay_alpha=0.4,
        simba=False,
        log_interval=200,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
    ):
        super().__init__(
            env_builder,
            model_builder_maker,
            num_workers,
            eval_eps,
            gamma,
            learning_rate,
            buffer_size,
            train_freq,
            gradient_steps,
            batch_size,
            1,
            learning_starts,
            0,
            True,
            prioritized_replay_alpha,
            0,
            0,
            scaled_by_reset,
            simba,
            log_interval,
            log_dir,
            _init_setup_model,
            policy_kwargs,
            full_tensorboard_log,
            seed,
            optimizer,
        )

        self.name = "TD7"
        self.action_noise = action_noise
        self.target_action_noise = action_noise * target_action_noise_mul
        self.action_noise_clamp = 0.5  # self.target_action_noise*1.5
        self.target_network_update_freq = target_network_update_freq
        self.policy_delay = policy_delay

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        self.checkpointing = False
        self.steps_before_checkpointing = int(5e5)
        self.max_eps_before_checkpointing = 20

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.policy_kwargs,
        )
        (
            self.preproc,
            self.encoder,
            self.action_encoder,
            self.actor,
            self.critic,
            self.encoder_params,
            self.policy_params,
            self.critic_params,
        ) = model_builder(next(self.key_seq), print_model=True)
        self.fixed_encoder_params = deepcopy(self.encoder_params)
        self.fixed_encoder_target_params = deepcopy(self.encoder_params)
        self.target_policy_params = deepcopy(self.policy_params)
        self.target_critic_params = deepcopy(self.critic_params)
        self.checkpoint_encoder_params = deepcopy(self.encoder_params)
        self.checkpoint_policy_params = deepcopy(self.policy_params)

        self.critic_params["values"] = {
            "min_value": jnp.array([np.inf], dtype=jnp.float32),
            "max_value": jnp.array([-np.inf], dtype=jnp.float32),
        }
        self.target_critic_params["values"] = {
            "min_value": jnp.array([0], dtype=jnp.float32),
            "max_value": jnp.array([0], dtype=jnp.float32),
        }

        self.encoder_opt_state = self.optimizer.init(self.encoder_params)
        self.opt_policy_state = self.optimizer.init(self.policy_params)
        self.opt_critic_state = self.optimizer.init(self.critic_params)
        self._get_actions = jax.jit(self._get_actions)
        self._train_step = jax.jit(self._train_step)

    def _get_actions(self, encoder_params, policy_params, obses, key=None) -> jnp.ndarray:
        feature = self.preproc(encoder_params, key, convert_jax(obses))
        zs = self.encoder(encoder_params, key, feature)
        return self.actor(policy_params, key, feature, zs)

    def actions(self, obs, steps, use_checkpoint=False, exploration=True):
        if self.simba:
            if exploration:
                self.obs_rms.update(obs)
            obs = self.obs_rms.normalize(obs)

        if self.learning_starts < steps:
            if use_checkpoint:
                actions = np.asarray(
                    self._get_actions(
                        self.checkpoint_encoder_params, self.checkpoint_policy_params, obs, None
                    )
                )
            else:
                actions = np.asarray(
                    self._get_actions(self.fixed_encoder_params, self.policy_params, obs, None)
                )
            if exploration:
                actions = np.clip(
                    actions
                    + self.action_noise
                    * np.random.normal(0, 1, size=(self.worker_size, self.action_size[0])),
                    -1,
                    1,
                )
        else:
            actions = np.random.uniform(-1.0, 1.0, size=(self.worker_size, self.action_size[0]))
        return actions

    def end_episode(self, steps, score, eplen):
        if self.learning_starts > steps:
            return
        self.eps_since_update += 1
        self.timesteps_since_update += eplen

        self.min_return = min(self.min_return, score)
        # End evaluation of current policy early
        if self.min_return < self.best_min_return:
            self.train_and_reset(steps)

        # Update checkpoint
        elif self.eps_since_update >= self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_policy_params = self.policy_params
            self.checkpoint_encoder_params = self.fixed_encoder_params
            self.train_and_reset(steps)

    def train_and_reset(self, steps):
        if not self.checkpointing and steps > self.steps_before_checkpointing:
            self.best_min_return = 0.9 * self.best_min_return
            self.max_eps_before_update = self.max_eps_before_checkpointing
            self.checkpointing = True
        self.loss_mean = self.train_step(steps, self.timesteps_since_update * self.gradient_steps)

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8

    def train_step(self, steps, gradient_steps):
        # Sample a batch from the replay buffer
        repr_losses = []
        losses = []
        targets = []
        for _ in range(gradient_steps):
            self.train_steps_count += 1
            if self.prioritized_replay:
                data = self.replay_buffer.sample(self.batch_size, self.prioritized_replay_beta0)
            else:
                data = self.replay_buffer.sample(self.batch_size)

            if self.simba:
                data["obses"] = self.obs_rms.normalize(data["obses"])
                data["nxtobses"] = self.obs_rms.normalize(data["nxtobses"])

            (
                self.encoder_params,
                self.policy_params,
                self.critic_params,
                self.fixed_encoder_params,
                self.fixed_encoder_target_params,
                self.target_policy_params,
                self.target_critic_params,
                self.encoder_opt_state,
                self.opt_policy_state,
                self.opt_critic_state,
                repr_loss,
                loss,
                t_mean,
                new_priorities,
            ) = self._train_step(
                self.encoder_params,
                self.policy_params,
                self.critic_params,
                self.fixed_encoder_params,
                self.fixed_encoder_target_params,
                self.target_policy_params,
                self.target_critic_params,
                self.encoder_opt_state,
                self.opt_policy_state,
                self.opt_critic_state,
                next(self.key_seq),
                self.train_steps_count,
                **data,
            )
            repr_losses.append(repr_loss)
            losses.append(loss)
            targets.append(t_mean)

            if self.prioritized_replay:
                self.replay_buffer.update_priorities(data["indexes"], new_priorities)

        mean_repr_loss = jnp.mean(jnp.array(repr_losses))
        mean_loss = jnp.mean(jnp.array(losses))
        mean_target = jnp.mean(jnp.array(targets))

        if self.logger_run:
            self.logger_run.log_metric("loss/encoder_loss", mean_repr_loss, steps)
            self.logger_run.log_metric("loss/qloss", mean_loss, steps)
            self.logger_run.log_metric("loss/targets", mean_target, steps)
            self.logger_run.log_metric(
                "loss/min_value", self.critic_params["values"]["min_value"], steps
            )
            self.logger_run.log_metric(
                "loss/max_value", self.critic_params["values"]["max_value"], steps
            )

        return mean_loss

    def _train_step(
        self,
        encoder_params,
        policy_params,
        critic_params,
        fixed_encoder_params,
        fixed_encoder_target_params,
        target_policy_params,
        target_critic_params,
        encoder_opt_state,
        opt_policy_state,
        opt_critic_state,
        key,
        step,
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
        not_terminateds = 1.0 - terminateds

        repr_loss, grad = jax.value_and_grad(self._encoder_loss)(
            encoder_params, obses, nxtobses, actions, key
        )
        updates, encoder_opt_state = self.optimizer.update(
            grad, encoder_opt_state, params=encoder_params
        )
        encoder_params = optax.apply_updates(encoder_params, updates)

        targets = self._target(
            fixed_encoder_target_params,
            target_policy_params,
            target_critic_params,
            rewards,
            nxtobses,
            not_terminateds,
            key,
        )
        critic_params["values"]["min_value"] = jnp.minimum(
            jnp.min(targets), critic_params["values"]["min_value"]
        )
        critic_params["values"]["max_value"] = jnp.maximum(
            jnp.max(targets), critic_params["values"]["max_value"]
        )

        feature, zs = self.feature_and_zs(fixed_encoder_params, obses, key)

        (critic_loss, priority), grad = jax.value_and_grad(self._critic_loss, has_aux=True)(
            critic_params, fixed_encoder_params, feature, zs, actions, targets, key
        )
        updates, opt_critic_state = self.optimizer.update(
            grad, opt_critic_state, params=critic_params
        )
        critic_params = optax.apply_updates(critic_params, updates)

        def _opt_actor(policy_params, opt_policy_state, key):
            grad = jax.grad(self._actor_loss)(
                policy_params, critic_params, fixed_encoder_params, feature, zs, key
            )
            updates, opt_policy_state = self.optimizer.update(
                grad, opt_policy_state, params=policy_params
            )
            policy_params = optax.apply_updates(policy_params, updates)
            return policy_params, opt_policy_state, key

        policy_params, opt_policy_state, key = jax.lax.cond(
            step % self.policy_delay == 0,
            lambda x: _opt_actor(*x),
            lambda x: x,
            (policy_params, opt_policy_state, key),
        )

        target_policy_params = hard_update(
            policy_params, target_policy_params, step, self.target_network_update_freq
        )
        target_critic_params = hard_update(
            critic_params, target_critic_params, step, self.target_network_update_freq
        )
        fixed_encoder_target_params = hard_update(
            fixed_encoder_params, fixed_encoder_target_params, step, self.target_network_update_freq
        )
        fixed_encoder_params = hard_update(
            encoder_params, fixed_encoder_params, step, self.target_network_update_freq
        )
        if self.scaled_by_reset:
            policy_params = scaled_by_reset(
                policy_params,
                opt_policy_state,
                self.optimizer,
                key,
                step,
                self.reset_freq,
                0.1,  # tau = 0.1 is softreset, but original paper uses 1.0
            )
            critic_params = scaled_by_reset(
                critic_params,
                opt_critic_state,
                self.optimizer,
                key,
                step,
                self.reset_freq,
                0.1,  # tau = 0.1 is softreset, but original paper uses 1.0
            )
        return (
            encoder_params,
            policy_params,
            critic_params,
            fixed_encoder_params,
            fixed_encoder_target_params,
            target_policy_params,
            target_critic_params,
            encoder_opt_state,
            opt_policy_state,
            opt_critic_state,
            repr_loss,
            critic_loss,
            jnp.mean(targets),
            priority,
        )

    def feature_and_zs(self, encoder_params, obses, key):
        feature = self.preproc(encoder_params, key, convert_jax(obses))
        zs = self.encoder(encoder_params, key, feature)
        return feature, zs

    def _encoder_loss(self, encoder_params, obses, next_obses, actions, key):
        next_zs = jax.lax.stop_gradient(
            self.encoder(encoder_params, key, self.preproc(encoder_params, key, next_obses))
        )
        zs = self.encoder(encoder_params, key, self.preproc(encoder_params, key, obses))
        pred_zs = self.action_encoder(encoder_params, key, zs, actions)
        loss = jnp.mean(jnp.square(next_zs - pred_zs))
        return loss

    def _actor_loss(self, policy_params, critic_params, fixed_encoder_params, feature, zs, key):
        actions = self.actor(policy_params, key, feature, zs)
        zsa = self.action_encoder(fixed_encoder_params, key, zs, actions)
        q1, q2 = self.critic(critic_params, key, feature, zs, zsa, actions)
        return -jnp.mean(jnp.minimum(q1, q2))

    def _critic_loss(self, critic_params, fixed_encoder_params, feature, zs, actions, targets, key):
        zsa = self.action_encoder(fixed_encoder_params, key, zs, actions)

        q1, q2 = self.critic(critic_params, key, feature, zs, zsa, actions)
        error1 = jnp.squeeze(q1 - targets)
        error2 = jnp.squeeze(q2 - targets)
        critic_loss = jnp.mean(hubberloss(error1, 1.0)) + jnp.mean(hubberloss(error2, 1.0))

        priority = jnp.maximum(jnp.maximum(jnp.abs(error1), jnp.abs(error2)), 1.0)
        return critic_loss, priority

    def _target(
        self,
        fixed_encoder_target_params,
        target_policy_params,
        target_critic_params,
        rewards,
        nxtobses,
        not_terminateds,
        key,
    ):
        next_feature = self.preproc(fixed_encoder_target_params, key, nxtobses)
        fixed_target_zs = self.encoder(fixed_encoder_target_params, key, next_feature)

        next_action = jnp.clip(
            self.actor(target_policy_params, key, next_feature, fixed_target_zs)
            + jnp.clip(
                self.target_action_noise
                * jax.random.normal(key, (self.batch_size, self.action_size[0])),
                -self.action_noise_clamp,
                self.action_noise_clamp,
            ),
            -1.0,
            1.0,
        )

        fixed_target_zsa = self.action_encoder(
            fixed_encoder_target_params, key, fixed_target_zs, next_action
        )

        q1, q2 = self.critic(
            target_critic_params, key, next_feature, fixed_target_zs, fixed_target_zsa, next_action
        )
        next_q = jnp.clip(
            jnp.minimum(q1, q2),
            target_critic_params["values"]["min_value"],
            target_critic_params["values"]["max_value"],
        )
        return rewards + not_terminateds * self.gamma * next_q

    def discription(self, eval_result=None):
        discription = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                discription += f"{k} : {v:8.2f}, "

        discription += f"loss : {np.mean(self.loss_mean):.3f}"
        return discription

    def run_name_update(self, run_name):
        if self.simba:
            run_name = "Simba_" + run_name
        if self.n_step_method:
            run_name = "{}Step_".format(self.n_step) + run_name
        return run_name

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="TD7",
        run_name="TD7",
    ):
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            experiment_name,
            run_name,
        )

    def learn_SingleEnv(self, pbar, callback=None, log_interval=1000):
        obs, info = self.env.reset()
        obs = [np.expand_dims(obs, axis=0)]
        self.scores = np.zeros([self.worker_size])
        self.eplen = np.zeros([self.worker_size], dtype=np.int32)
        self.score_mean = None
        self.loss_mean = None
        for steps in pbar:
            self.eplen += 1
            actions = self.actions(obs, steps)
            next_obs, reward, terminated, truncated, info = self.env.step(actions[0])
            next_obs = [np.expand_dims(next_obs, axis=0)]
            self.replay_buffer.add(obs, actions[0], reward, next_obs, terminated, truncated)
            self.scores[0] += reward
            obs = next_obs
            if terminated or truncated:
                self.end_episode(steps, self.scores[0], self.eplen[0])
                self.scores[0] = 0
                self.eplen[0] = 0
                obs, info = self.env.reset()
                obs = [np.expand_dims(obs, axis=0)]

            if steps % self.eval_freq == 0:
                eval_result = self.eval(steps)

            if steps % log_interval == 0 and eval_result is not None and self.loss_mean is not None:
                pbar.set_description(self.discription(eval_result))

    def eval(self, steps):
        total_reward = np.zeros(self.eval_eps)
        total_ep_len = np.zeros(self.eval_eps)
        total_truncated = np.zeros(self.eval_eps)
        for ep in range(self.eval_eps):
            obs, info = self.eval_env.reset()
            obs = [np.expand_dims(obs, axis=0)]
            terminated = False
            truncated = False
            eplen = 0
            while not terminated and not truncated:
                actions = self.actions(obs, steps, use_checkpoint=True, exploration=False)
                next_obs, reward, terminated, truncated, info = self.eval_env.step(actions[0])
                next_obs = [np.expand_dims(next_obs, axis=0)]
                # self.replay_buffer.add(obs, actions[0], reward, next_obs, terminated, truncated)
                total_reward[ep] += reward
                obs = next_obs
                eplen += 1
            total_ep_len[ep] = eplen
            total_truncated[ep] = float(truncated)

        mean_reward = np.mean(total_reward)
        mean_ep_len = np.mean(total_ep_len)

        if self.logger_run:
            self.logger_run.log_metric("env/episode_reward", mean_reward, steps)
            self.logger_run.log_metric("env/episode len", mean_ep_len, steps)
            self.logger_run.log_metric("env/time over", np.mean(total_truncated), steps)
        return {"mean_reward": mean_reward, "mean_ep_len": mean_ep_len}
