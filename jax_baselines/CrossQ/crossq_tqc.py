import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.common.losses import QuantileHuberLosses
from jax_baselines.common.utils import convert_jax, scaled_by_reset, truncated_mixture
from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family


class CrossQ_TQC(Deteministic_Policy_Gradient_Family):
    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        num_workers=1,
        eval_eps=20,
        gamma=0.995,
        learning_rate=3e-4,
        buffer_size=100000,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        n_support=25,
        delta=1.0,
        critic_num=2,
        quantile_drop=0.05,
        batch_size=32,
        n_step=1,
        learning_starts=1000,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-6,
        mixture_type="truncated",
        risk_avoidance=0.0,
        scaled_by_reset=False,
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
            n_step,
            learning_starts,
            0,  # target_network_update_tau
            prioritized_replay,
            prioritized_replay_alpha,
            prioritized_replay_beta0,
            prioritized_replay_eps,
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

        self.name = "CrossQ_TQC"
        self._ent_coef = ent_coef
        self.target_entropy = 0.5 * np.prod(self.action_size).astype(
            np.float32
        )  # -np.sqrt(np.prod(self.action_size).astype(np.float32))
        self.ent_coef_learning_rate = 1e-4
        self.n_support = n_support
        self.delta = delta
        self.critic_num = critic_num
        self.quantile_drop = int(max(np.round(self.critic_num * self.n_support * quantile_drop), 1))
        self.middle_support = int(np.floor(n_support / 2.0))
        self.mixture_type = mixture_type
        self.risk_avoidance = risk_avoidance

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.n_support,
            self.policy_kwargs,
        )
        (
            self.preproc,
            self.actor,
            self.critic,
            self.policy_params,
            self.critic_params,
        ) = model_builder(next(self.key_seq), print_model=True)
        self.opt_policy_state = self.optimizer.init(self.policy_params)
        self.opt_critic_state = self.optimizer.init(self.critic_params)

        if isinstance(self._ent_coef, str) and self._ent_coef.startswith("auto"):
            init_value = np.log(1e-1)
            if "_" in self._ent_coef:
                init_value = np.log(float(self._ent_coef.split("_")[1]))
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"
            self.log_ent_coef = jax.device_put(init_value)
            self.auto_entropy = True
        else:
            try:
                self.log_ent_coef = jnp.log(float(self._ent_coef))
            except ValueError:
                raise ValueError("Invalid value for ent_coef: {}".format(self._ent_coef))
            self.auto_entropy = False

        self.quantile = (
            jnp.linspace(0.0, 1.0, self.n_support + 1, dtype=jnp.float32)[1:]
            + jnp.linspace(0.0, 1.0, self.n_support + 1, dtype=jnp.float32)[:-1]
        ) / 2.0  # [support]
        self.quantile = jax.device_put(jnp.expand_dims(self.quantile, axis=(0, 1))).astype(
            jnp.float32
        )  # [1 x 1 x support]

        self._get_actions = jax.jit(self._get_actions)
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

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        mu, log_std = self.actor(params, None, self.preproc(params, None, convert_jax(obses)))
        std = jnp.exp(log_std)
        pi = jax.nn.tanh(mu + std * jax.random.normal(key, std.shape))
        return pi

    def actions(self, obs, steps, eval=False):
        if self.simba:
            if steps != np.inf:
                self.obs_rms.update(obs)
            obs = self.obs_rms.normalize(obs)

        if self.learning_starts < steps:
            actions = np.asarray(self._get_actions(self.policy_params, obs, next(self.key_seq)))
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
                self.policy_params,
                self.critic_params,
                self.opt_policy_state,
                self.opt_critic_state,
                loss,
                t_mean,
                self.log_ent_coef,
                new_priorities,
            ) = self._train_step(
                self.policy_params,
                self.critic_params,
                self.opt_policy_state,
                self.opt_critic_state,
                next(self.key_seq),
                self.train_steps_count,
                self.log_ent_coef,
                **data
            )

            if self.prioritized_replay:
                self.replay_buffer.update_priorities(data["indexes"], new_priorities)

        if self.logger_run and steps % self.log_interval == 0:
            self.logger_run.log_metric("loss/qloss", loss, steps)
            self.logger_run.log_metric("loss/targets", t_mean, steps)
            self.logger_run.log_metric("loss/ent_coef", np.exp(self.log_ent_coef), steps)

        return loss

    def _train_step(
        self,
        policy_params,
        critic_params,
        opt_policy_state,
        opt_critic_state,
        key,
        step,
        log_ent_coef,
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
        key1, key2 = jax.random.split(key, 2)

        (critic_loss, (abs_error, critic_params)), grad = jax.value_and_grad(
            self._critic_loss, has_aux=True
        )(
            critic_params,
            policy_params,
            obses,
            actions,
            rewards,
            nxtobses,
            not_terminateds,
            ent_coef,
            weights,
            key1,
        )
        updates, opt_critic_state = self.optimizer.update(
            grad, opt_critic_state, params=critic_params
        )
        critic_params = optax.apply_updates(critic_params, updates)

        (actor_loss, log_prob), grad = jax.value_and_grad(self._actor_loss, has_aux=True)(
            policy_params, critic_params, obses, key2, ent_coef
        )
        updates, opt_policy_state = self.optimizer.update(
            grad, opt_policy_state, params=policy_params
        )
        policy_params = optax.apply_updates(policy_params, updates)

        if self.auto_entropy:
            log_ent_coef = self._train_ent_coef(log_ent_coef, log_prob)

        new_priorities = None
        if self.prioritized_replay:
            new_priorities = abs_error
        if self.scaled_by_reset:
            policy_params = scaled_by_reset(
                policy_params,
                key,
                step,
                self.reset_freq,
                0.1,  # tau = 0.1 is softreset, but original paper uses 1.0
            )
            critic_params = scaled_by_reset(
                critic_params,
                key,
                step,
                self.reset_freq,
                0.1,  # tau = 0.1 is softreset, but original paper uses 1.0
            )
        return (
            policy_params,
            critic_params,
            opt_policy_state,
            opt_critic_state,
            critic_loss,
            -actor_loss,
            log_ent_coef,
            new_priorities,
        )

    def _train_ent_coef(self, log_coef, log_prob):
        def loss(log_ent_coef, log_prob):
            ent_coef = jnp.exp(log_ent_coef)
            return jnp.mean(ent_coef * (self.target_entropy - log_prob))

        grad = jax.grad(loss)(log_coef, log_prob)
        log_coef = log_coef - self.ent_coef_learning_rate * grad
        return log_coef

    def _critic_loss(
        self,
        critic_params,
        policy_params,
        obses,
        actions,
        rewards,
        nxtobses,
        not_terminateds,
        ent_coef,
        weights,
        key,
    ):
        concated_obses = [jnp.concatenate([o, n]) for o, n in zip(obses, nxtobses)]
        concated_preproc = self.preproc(policy_params, key, concated_obses)
        next_preproc = jnp.split(concated_preproc, 2, axis=0)[1]
        next_policy, log_prob = self._get_pi_log_prob(policy_params, next_preproc, key)
        concated_actions = jnp.concatenate([actions, next_policy])
        (q1, q2), variable_updates = self.critic(
            critic_params, key, concated_preproc, concated_actions, True
        )
        critic_params["batch_stats"] = variable_updates["batch_stats"]
        q1, next_q1 = jnp.split(q1, 2, axis=0)
        q2, next_q2 = jnp.split(q2, 2, axis=0)
        if self.mixture_type == "min":
            next_q = jnp.min(jnp.stack((next_q1, next_q2), axis=-1), axis=-1) - ent_coef * log_prob
        elif self.mixture_type == "truncated":
            next_q = truncated_mixture((q1, q2), self.quantile_drop) - ent_coef * log_prob
        logit_valid_tile = jax.lax.stop_gradient((not_terminateds * next_q * self._gamma) + rewards)
        logit_valid_tile = jnp.expand_dims(logit_valid_tile, axis=2)  # batch x support x 1
        huber0 = QuantileHuberLosses(
            logit_valid_tile,
            jnp.expand_dims(q1, axis=1),
            self.quantile,
            self.delta,
        )
        huber1 = QuantileHuberLosses(
            logit_valid_tile,
            jnp.expand_dims(q2, axis=1),
            self.quantile,
            self.delta,
        )
        critic_loss = jnp.mean(weights * huber0) + jnp.mean(weights * huber1)
        return critic_loss, (huber0, critic_params)

    def _actor_loss(self, policy_params, critic_params, obses, key, ent_coef):
        feature = self.preproc(policy_params, key, obses)
        policy, log_prob = self._get_pi_log_prob(policy_params, feature, key)
        q_pis, _ = self.critic(critic_params, key, feature, policy, False)
        actor_loss = jnp.mean(
            ent_coef * log_prob - jnp.mean(jnp.concatenate(q_pis, axis=1), axis=1)
        )
        return actor_loss, log_prob

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="CrossQ_TQC",
        run_name="CrossQ_TQC",
    ):
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            experiment_name,
            run_name,
        )
