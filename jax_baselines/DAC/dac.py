from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.common.utils import convert_jax, scaled_by_reset, soft_update
from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family

DAC_COEF_GRAD_CLIP = 1e3


class DAC(Deteministic_Policy_Gradient_Family):
    def __init__(self, env_builder: callable, model_builder_maker, ent_coef="auto", **kwargs):
        super().__init__(env_builder, model_builder_maker, **kwargs)

        self.name = "DAC"
        self._ent_coef = ent_coef
        self.target_entropy = 0.5 * np.prod(self.action_size).astype(
            np.float32
        )  # -np.sqrt(np.prod(self.action_size).astype(np.float32))
        self.pessimism_coef = -0.2  # -0.2
        self.kl_target = 0.25
        self.ent_coef_learning_rate = 1e-4
        self.optimism_coef_learning_rate = 3e-5
        self.kl_weight_learning_rate = 3e-5

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.policy_kwargs,
        )
        (
            self.preproc,
            self.actor,
            self.optimistic_actor,
            self.critic,
            self.pessimistic_policy_params,
            self.optimistic_policy_params,
            self.critic_params,
        ) = model_builder(next(self.key_seq), print_model=True)
        self.target_critic_params = deepcopy(self.critic_params)
        self.opt_pessimistic_policy_state = self.optimizer.init(self.pessimistic_policy_params)
        self.opt_optimistic_policy_state = self.optimizer.init(self.optimistic_policy_params)
        self.opt_critic_state = self.optimizer.init(self.critic_params)

        if isinstance(self._ent_coef, str) and self._ent_coef.startswith("auto"):
            init_value = np.log(1e-1)
            if "_" in self._ent_coef:
                init_value = np.log(float(self._ent_coef.split("_")[1]))
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"
            self.log_ent_coef = jax.device_put(init_value)
            self.auto_entropy = True
            self.log_optimism_coef = jax.device_put(np.array(np.log(1.0)))
            self.log_kl_weight = jax.device_put(np.array(np.log(0.25)))
        else:
            try:
                self.log_ent_coef = jnp.log(float(self._ent_coef))
            except ValueError:
                raise ValueError("Invalid value for ent_coef: {}".format(self._ent_coef))
            self.auto_entropy = False

        self._get_actions = jax.jit(self._get_actions)
        self._get_actions_o = jax.jit(self._get_actions_o)
        self._train_step = jax.jit(self._train_step)
        self._train_ent_coef = jax.jit(self._train_ent_coef)

    def _get_pi_log_prob(self, params, feature, key=None) -> jnp.ndarray:
        mu, log_std = self.actor(params, None, feature)
        std = jnp.exp(log_std)
        x_t = mu + std * jax.random.normal(key, std.shape)
        pi = jax.nn.tanh(x_t)
        log_prob = jnp.sum(
            -0.5 * (jnp.square((x_t - mu) / (std + 1e-6)) + 2 * log_std + jnp.log(2 * np.pi))
            - jnp.log(1 - jnp.square(pi) + 1e-6),
            axis=1,
            keepdims=True,
        )
        return pi, log_prob

    def _get_pi_kl_divergence(
        self, pessimistic_policy_params, optimistic_policy_params, feature, key=None
    ):
        mu_p, log_std_p = self.actor(pessimistic_policy_params, None, feature)
        mu_o, log_std_o = self.optimistic_actor(
            optimistic_policy_params, None, mu_p, log_std_p, feature
        )
        std_p = jnp.exp(log_std_p)
        std_o = jnp.exp(log_std_o)
        std_o_bar = std_o / 1.25
        kl_divergence = jnp.sum(
            jnp.log(std_p / std_o_bar)
            + (jnp.square(std_o_bar) + jnp.square(mu_o - mu_p))
            / (2 * jnp.square(std_p))  # log(variance_p/variance_o)
            - 0.5,
            axis=1,
            keepdims=True,
        )
        x_t = mu_o + std_o * jax.random.normal(key, mu_o.shape)
        pi_o = jax.nn.tanh(x_t)
        return pi_o, kl_divergence

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        mu, log_std = self.actor(params, None, self.preproc(params, None, convert_jax(obses)))
        std = jnp.exp(log_std)
        x_t = mu + std * jax.random.normal(key, std.shape)
        pi = jax.nn.tanh(x_t)
        return pi

    def _get_actions_o(self, params, optimistic_params, obses, key=None) -> jnp.ndarray:
        mu, log_std = self.actor(params, None, self.preproc(params, None, convert_jax(obses)))
        mu_optimistic, log_std_optimistic = self.optimistic_actor(
            optimistic_params,
            None,
            mu,
            log_std,
            self.preproc(optimistic_params, None, convert_jax(obses)),
        )
        std = jnp.exp(log_std_optimistic)
        x_t = mu_optimistic + std * jax.random.normal(key, std.shape)
        pi = jax.nn.tanh(x_t)
        return pi

    def actions(self, obs, steps, eval=False):
        if self.simba:
            if steps != np.inf:
                self.obs_rms.update(obs)
            obs = self.obs_rms.normalize(obs)

        if self.learning_starts < steps:
            if eval:
                actions = np.asarray(
                    self._get_actions(self.pessimistic_policy_params, obs, next(self.key_seq))
                )
            else:
                actions = np.asarray(
                    self._get_actions_o(
                        self.pessimistic_policy_params,
                        self.optimistic_policy_params,
                        obs,
                        next(self.key_seq),
                    )
                )
        else:
            actions = np.random.uniform(-1.0, 1.0, size=(self.worker_size, self.action_size[0]))
        return actions

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
                self.pessimistic_policy_params,
                self.optimistic_policy_params,
                self.critic_params,
                self.target_critic_params,
                self.opt_pessimistic_policy_state,
                self.opt_optimistic_policy_state,
                self.opt_critic_state,
                loss,
                t_mean,
                kl_divergence,
                self.log_ent_coef,
                self.log_optimism_coef,
                self.log_kl_weight,
                new_priorities,
            ) = self._train_step(
                self.pessimistic_policy_params,
                self.optimistic_policy_params,
                self.critic_params,
                self.target_critic_params,
                self.opt_pessimistic_policy_state,
                self.opt_optimistic_policy_state,
                self.opt_critic_state,
                next(self.key_seq),
                self.train_steps_count,
                self.log_ent_coef,
                self.log_optimism_coef,
                self.log_kl_weight,
                **data
            )

            if self.prioritized_replay:
                self.replay_buffer.update_priorities(data["indexes"], new_priorities)

        if self.logger_run and steps % self.log_interval == 0:
            self.logger_run.log_metric("loss/qloss", loss, steps)
            self.logger_run.log_metric("loss/targets", t_mean, steps)
            self.logger_run.log_metric("loss/kl_divergence", kl_divergence, steps)
            self.logger_run.log_metric("loss/ent_coef", np.exp(self.log_ent_coef), steps)
            self.logger_run.log_metric("loss/optimism_coef", np.exp(self.log_optimism_coef), steps)
            self.logger_run.log_metric("loss/kl_weight", np.exp(self.log_kl_weight), steps)

        return loss

    def _train_step(
        self,
        pessimistic_policy_params,
        optimistic_policy_params,
        critic_params,
        target_critic_params,
        opt_pessimistic_policy_state,
        opt_optimistic_policy_state,
        opt_critic_state,
        key,
        step,
        log_ent_coef,
        log_optimism_coef,
        log_kl_weight,
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
        ent_coef = jnp.exp(log_ent_coef)
        optimism_coef = jnp.exp(log_optimism_coef)
        kl_weight = jnp.exp(log_kl_weight)
        key1, key2, key3, key4 = jax.random.split(key, 4)
        targets = self._target(
            pessimistic_policy_params,
            target_critic_params,
            rewards,
            nxtobses,
            not_terminateds,
            key1,
            ent_coef,
        )

        (critic_loss, abs_error), grad = jax.value_and_grad(self._critic_loss, has_aux=True)(
            critic_params, pessimistic_policy_params, obses, actions, targets, weights, key2
        )
        updates, opt_critic_state = self.optimizer.update(
            grad, opt_critic_state, params=critic_params
        )
        critic_params = optax.apply_updates(critic_params, updates)

        (actor_loss, log_prob), grad = jax.value_and_grad(
            self._pessimistic_actor_loss, has_aux=True
        )(pessimistic_policy_params, critic_params, obses, key3, ent_coef)
        updates, opt_pessimistic_policy_state = self.optimizer.update(
            grad, opt_pessimistic_policy_state, params=pessimistic_policy_params
        )
        pessimistic_policy_params = optax.apply_updates(pessimistic_policy_params, updates)

        (actor_loss, kl_divergence), grad = jax.value_and_grad(
            self._optimistic_actor_loss, has_aux=True
        )(
            optimistic_policy_params,
            pessimistic_policy_params,
            critic_params,
            obses,
            key4,
            optimism_coef,
            kl_weight,
        )
        updates, opt_optimistic_policy_state = self.optimizer.update(
            grad, opt_optimistic_policy_state, params=optimistic_policy_params
        )
        optimistic_policy_params = optax.apply_updates(optimistic_policy_params, updates)

        if self.auto_entropy:
            log_ent_coef = self._train_ent_coef(log_ent_coef, log_prob)
            # log_optimism_coef = self._train_optimism_coef(log_optimism_coef, kl_divergence)
            # log_kl_weight = self._train_kl_weight(log_kl_weight, kl_divergence)

        target_critic_params = soft_update(
            critic_params, target_critic_params, self.target_network_update_tau
        )

        new_priorities = None
        if self.prioritized_replay:
            new_priorities = abs_error
        if self.scaled_by_reset:
            pessimistic_policy_params = scaled_by_reset(
                pessimistic_policy_params,
                opt_pessimistic_policy_state,
                self.optimizer,
                key,
                step,
                self.reset_freq,
                0.1,  # tau = 0.1 is softreset, but original paper uses 1.0
            )
            optimistic_policy_params = scaled_by_reset(
                optimistic_policy_params,
                opt_optimistic_policy_state,
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
            pessimistic_policy_params,
            optimistic_policy_params,
            critic_params,
            target_critic_params,
            opt_pessimistic_policy_state,
            opt_optimistic_policy_state,
            opt_critic_state,
            critic_loss,
            jnp.mean(targets),
            jnp.mean(kl_divergence),
            log_ent_coef,
            log_optimism_coef,
            log_kl_weight,
            new_priorities,
        )

    def _train_ent_coef(self, log_coef, log_prob):
        def loss(log_ent_coef, log_prob):
            ent_coef = jnp.exp(log_ent_coef)
            return jnp.mean(ent_coef * (self.target_entropy - log_prob))

        grad = jax.grad(loss)(log_coef, log_prob)
        log_coef = log_coef - self.ent_coef_learning_rate * grad
        return log_coef

    def _train_optimism_coef(self, log_optimism_coef, kl_divergence):
        def loss(log_optimism_coef, kl_divergence):
            optimism_coef = jnp.exp(log_optimism_coef)
            return jnp.mean(
                (optimism_coef + self.pessimism_coef)
                * (jnp.mean(kl_divergence) / self.action_size[0] - self.kl_target)
            )

        grad = jax.grad(loss)(log_optimism_coef, kl_divergence)
        log_optimism_coef = jnp.maximum(
            log_optimism_coef - self.optimism_coef_learning_rate * grad,
            jnp.log(self.pessimism_coef),
        )
        return log_optimism_coef

    def _train_kl_weight(self, log_kl_weight, kl_divergence):
        def loss(log_kl_weight, kl_divergence):
            kl_weight = jnp.exp(log_kl_weight)
            return -jnp.mean(
                kl_weight * (jnp.mean(kl_divergence) / self.action_size[0] - self.kl_target)
            )

        grad = jax.grad(loss)(log_kl_weight, kl_divergence)
        log_kl_weight = log_kl_weight - self.kl_weight_learning_rate * grad
        return log_kl_weight

    def _critic_loss(
        self, critic_params, pessimistic_policy_params, obses, actions, targets, weights, key
    ):
        feature = self.preproc(pessimistic_policy_params, key, obses)
        q1, q2 = self.critic(critic_params, key, feature, actions)
        error1 = jnp.squeeze(q1 - targets)
        error2 = jnp.squeeze(q2 - targets)
        critic_loss = jnp.mean(weights * jnp.square(error1)) + jnp.mean(
            weights * jnp.square(error2)
        )
        return critic_loss, jnp.abs(error1)

    def _pessimistic_actor_loss(
        self, pessimistic_policy_params, critic_params, obses, key, ent_coef
    ):
        feature = self.preproc(pessimistic_policy_params, key, obses)
        policy, log_prob = self._get_pi_log_prob(pessimistic_policy_params, feature, key)
        q1_pi, q2_pi = self.critic(critic_params, key, feature, policy)
        mean_q = (q1_pi + q2_pi) / 2.0
        std_q = jnp.abs(q1_pi - q2_pi) / 2.0
        actor_loss = jnp.mean(ent_coef * log_prob - (mean_q + self.pessimism_coef * std_q))
        return actor_loss, log_prob

    def _optimistic_actor_loss(
        self,
        optimistic_policy_params,
        pessimistic_policy_params,
        critic_params,
        obses,
        key,
        optimism_coef,
        kl_weight,
    ):
        feature = self.preproc(pessimistic_policy_params, key, obses)
        policy, kl_divergence = self._get_pi_kl_divergence(
            pessimistic_policy_params, optimistic_policy_params, feature, key
        )
        q1_pi, q2_pi = self.critic(critic_params, key, feature, policy)
        mean_q = (q1_pi + q2_pi) / 2.0
        std_q = jnp.abs(q1_pi - q2_pi) / 2.0
        actor_loss = jnp.mean(kl_weight * kl_divergence - (mean_q + optimism_coef * std_q))
        return actor_loss, kl_divergence

    def _target(
        self,
        pessimistic_policy_params,
        target_critic_params,
        rewards,
        nxtobses,
        not_terminateds,
        key,
        ent_coef,
    ):
        policy, log_prob = self._get_pi_log_prob(
            pessimistic_policy_params, self.preproc(pessimistic_policy_params, key, nxtobses), key
        )
        q1_pi, q2_pi = self.critic(
            target_critic_params,
            key,
            self.preproc(pessimistic_policy_params, key, nxtobses),
            policy,
        )
        mean_q = (q1_pi + q2_pi) / 2.0
        std_q = jnp.abs(q1_pi - q2_pi) / 2.0
        next_q = mean_q + self.pessimism_coef * std_q - ent_coef * log_prob
        return (not_terminateds * next_q * self._gamma) + rewards

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="DAC",
        run_name="DAC",
    ):
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            experiment_name,
            run_name,
        )
