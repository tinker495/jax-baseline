from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.common.schedules import LinearSchedule
from jax_baselines.common.utils import convert_jax, scaled_by_reset, soft_update
from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from jax_baselines.DDPG.ou_noise import OUNoise


class DDPG(Deteministic_Policy_Gradient_Family):
    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        exploration_initial_eps=1.0,
        **kwargs,
    ):

        self.name = "DDPG"
        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction

        super().__init__(env_builder, model_builder_maker, **kwargs)

        self.noise = OUNoise(action_size=self.action_size[0], worker_size=self.worker_size)

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.policy_kwargs,
        )
        (
            self.preproc,
            self.actor,
            self.critic,
            self.policy_params,
            self.critic_params,
        ) = model_builder(next(self.key_seq), print_model=True)
        self.target_policy_params = deepcopy(self.policy_params)
        self.target_critic_params = deepcopy(self.critic_params)

        self.opt_policy_state = self.optimizer.init(self.policy_params)
        self.opt_critic_state = self.optimizer.init(self.critic_params)
        self._get_actions = jax.jit(self._get_actions)
        self._train_step = jax.jit(self._train_step)

    def _get_actions(self, policy_params, obses, key=None) -> jnp.ndarray:
        return self.actor(
            policy_params, key, self.preproc(policy_params, key, convert_jax(obses))
        )  #

    def discription(self, eval_result=None):
        discription = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                discription += f"{k} : {v:8.2f}, "

        discription += f"loss : {np.mean(self.lossque):.3f}"
        discription += f"epsilon : {self.epsilon:.3f}"
        return discription

    def actions(self, obs, steps, eval=False):
        if self.simba:
            # During eval with checkpointing, normalize using snapshot obs_rms if available
            rms = (
                self.checkpoint_obs_rms
                if (eval and self.use_checkpointing and hasattr(self, "checkpoint_obs_rms"))
                else self.action_obs_rms
                if hasattr(self, "action_obs_rms")
                else self.obs_rms
            )
            # Only update live obs_rms during training (not eval) and when steps is finite
            if (not eval) and steps != np.inf:
                self.obs_rms.update(obs)
            obs = rms.normalize(obs)

        if self.learning_starts < steps:
            # Select params: during eval with checkpointing prefer snapshot
            policy_params = (
                self.checkpoint_policy_params
                if (
                    eval
                    and self.use_checkpointing
                    and self.checkpointing_enabled
                    and hasattr(self, "checkpoint_policy_params")
                )
                else self.policy_params
            )

            actions = np.asarray(self._get_actions(policy_params, obs, None))
            if not eval:
                self.epsilon = self.exploration.value(steps)
                actions = np.clip(actions + self.noise() * self.epsilon, -1, 1)
        else:
            actions = np.random.uniform(-1.0, 1.0, size=(self.worker_size, self.action_size[0]))
        return actions

    def test_action(self, obs):
        return np.clip(
            np.asarray(self._get_actions(self.policy_params, obs, None))
            + self.noise() * self.exploration_final_eps,
            -1,
            1,
        )

    def train_step(self, steps, gradient_steps):
        # Sample a batch from the replay buffer
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
                self.policy_params,
                self.critic_params,
                self.target_policy_params,
                self.target_critic_params,
                self.opt_policy_state,
                self.opt_critic_state,
                loss,
                t_mean,
                new_priorities,
            ) = self._train_step(
                self.policy_params,
                self.critic_params,
                self.target_policy_params,
                self.target_critic_params,
                self.opt_policy_state,
                self.opt_critic_state,
                self.train_steps_count,
                None,
                **data,
            )

            if self.prioritized_replay:
                self.replay_buffer.update_priorities(data["indexes"], new_priorities)

        if self.logger_run and (steps - self._last_log_step >= self.log_interval):
            self._last_log_step = steps
            self.logger_run.log_metric("loss/qloss", loss, steps)
            self.logger_run.log_metric("loss/targets", t_mean, steps)

        return loss

    def _train_step(
        self,
        policy_params,
        critic_params,
        target_policy_params,
        target_critic_params,
        opt_policy_state,
        opt_critic_state,
        step,
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
        not_terminateds = 1.0 - terminateds

        targets = self._target(
            target_policy_params, target_critic_params, rewards, nxtobses, not_terminateds, key
        )
        (critic_loss, abs_error), grad = jax.value_and_grad(self._critic_loss, has_aux=True)(
            critic_params, policy_params, obses, actions, targets, weights, key
        )
        updates, opt_critic_state = self.optimizer.update(
            grad, opt_critic_state, params=critic_params
        )
        critic_params = optax.apply_updates(critic_params, updates)

        grad = jax.grad(self._actor_loss)(policy_params, critic_params, obses, key)
        updates, opt_policy_state = self.optimizer.update(
            grad, opt_policy_state, params=policy_params
        )
        policy_params = optax.apply_updates(policy_params, updates)

        target_critic_params = soft_update(
            critic_params, target_critic_params, self.target_network_update_tau
        )
        target_policy_params = soft_update(
            policy_params, target_policy_params, self.target_network_update_tau
        )
        new_priorities = None
        if self.prioritized_replay:
            new_priorities = abs_error
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
            policy_params,
            critic_params,
            target_policy_params,
            target_critic_params,
            opt_policy_state,
            opt_critic_state,
            critic_loss,
            jnp.mean(targets),
            new_priorities,
        )

    def _critic_loss(self, critic_params, policy_params, obses, actions, targets, weights, key):
        feature = self.preproc(policy_params, key, obses)
        vals = self.critic(critic_params, key, feature, actions)
        error = jnp.squeeze(vals - targets)
        critic_loss = jnp.mean(weights * jnp.square(error))
        return critic_loss, jnp.abs(error)

    def _actor_loss(self, policy_params, critic_params, obses, key):
        feature = self.preproc(policy_params, key, obses)
        actions = self.actor(policy_params, key, feature)
        q = self.critic(critic_params, key, feature, actions)
        return -jnp.mean(q)

    def _target(
        self, target_policy_params, target_critic_params, rewards, nxtobses, not_terminateds, key
    ):
        next_feature = self.preproc(target_policy_params, key, nxtobses)
        next_action = self.actor(target_policy_params, key, next_feature)
        next_q = self.critic(target_critic_params, key, next_feature, next_action)
        return (not_terminateds * next_q * self._gamma) + rewards

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="DDPG",
        run_name="DDPG",
    ):
        self.exploration = LinearSchedule(
            schedule_timesteps=int(self.exploration_fraction * total_timesteps),
            initial_p=self.exploration_initial_eps,
            final_p=self.exploration_final_eps,
        )
        self.epsilon = 1.0
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            experiment_name,
            run_name,
        )
