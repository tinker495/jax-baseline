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
        target_action_noise_mul=2.0,
        action_noise=0.1,
        policy_delay=2,
        target_network_update_freq=250,
        **kwargs,
    ):
        # Set TD7-specific defaults - always enable checkpointing
        td7_kwargs = {
            "n_step": 1,
            "target_network_update_tau": 0,
            "prioritized_replay": True,
            "prioritized_replay_beta0": 0,
            "prioritized_replay_eps": 0,
            "use_checkpointing": True,  # TD7 always uses checkpointing
            **kwargs,
        }

        super().__init__(env_builder, model_builder_maker, **td7_kwargs)

        self.name = "TD7"
        self.action_noise = action_noise
        self.target_action_noise = action_noise * target_action_noise_mul
        self.action_noise_clamp = 0.5  # self.target_action_noise*1.5
        self.target_network_update_freq = target_network_update_freq
        self.policy_delay = policy_delay

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
            # Use checkpoint state during evaluation, current state during training
            if eval and self.use_checkpointing and hasattr(self, "checkpoint_state"):
                encoder_params = self.checkpoint_state.get("encoder")
                policy_params = self.checkpoint_state.get("policy")
            else:
                behavior_state = self.get_behavior_state()
                encoder_params = behavior_state.get("encoder")
                policy_params = behavior_state.get("policy")

            actions = np.asarray(self._get_actions(encoder_params, policy_params, obs, None))

            # Add exploration noise only during training
            if not eval:
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

        if self.logger_run and (steps - self._last_log_step >= self.log_interval):
            self._last_log_step = steps
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

        discription += f"loss : {np.mean(self.lossque):.3f}"
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
