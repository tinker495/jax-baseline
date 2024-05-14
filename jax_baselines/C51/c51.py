from copy import deepcopy

import jax
import jax.numpy as jnp
import optax

from jax_baselines.common.utils import convert_jax, hard_update, q_log_pi
from jax_baselines.DQN.base_class import Q_Network_Family


class C51(Q_Network_Family):
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
        train_freq=1,
        gradient_steps=1,
        batch_size=32,
        double_q=True,
        dueling_model=False,
        n_step=1,
        learning_starts=1000,
        target_network_update_freq=2000,
        prioritized_replay=False,
        prioritized_replay_alpha=0.9,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-3,
        param_noise=False,
        munchausen=False,
        log_interval=200,
        tensorboard_log=None,
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
            double_q,
            dueling_model,
            n_step,
            learning_starts,
            target_network_update_freq,
            prioritized_replay,
            prioritized_replay_alpha,
            prioritized_replay_beta0,
            prioritized_replay_eps,
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

        self.name = "C51"
        self.sigma = 0.75
        self.categorial_bar_n = categorial_bar_n
        self.categorial_max = float(categorial_max)
        self.categorial_min = float(categorial_min)

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs

        self.model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.dueling_model,
            self.param_noise,
            self.categorial_bar_n,
            self.policy_kwargs,
        )

        self.preproc, self.model, self.params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)

        self.support = jnp.linspace(
            self.categorial_min, self.categorial_max, self.categorial_bar_n + 1, dtype=jnp.float32
        )
        bin_width = self.support[1] - self.support[0]
        self.centers = (self.support[:-1] + self.support[1:]) / 2
        self._centers = jnp.expand_dims(self.centers, axis=0)
        self.sigma = self.sigma * bin_width

        self.get_q = jax.jit(self.get_q)
        self._get_actions = jax.jit(self._get_actions)
        self._loss = jax.jit(self._loss)
        self._target = jax.jit(self._target)
        self._train_step = jax.jit(self._train_step)

    def get_q(self, params, obses, key=None) -> jnp.ndarray:
        return self.model(params, key, self.preproc(params, key, obses))

    def to_probs(self, target: jax.Array, probs: jax.Array):
        # target: [batch, support]
        # probs: [batch, support]
        def f(target):
            cdf_evals = jax.scipy.special.erf((self.support - target) / (jnp.sqrt(2) * self.sigma))
            z = cdf_evals[-1] - cdf_evals[0]
            bin_probs = cdf_evals[1:] - cdf_evals[:-1]
            return bin_probs / z

        hl_gauss_probs = jax.vmap(jax.vmap(f))(target)  # [batch, support, support]
        probs = jnp.expand_dims(probs, axis=1) # [batch, 1, support]
        return jnp.sum(hl_gauss_probs * probs, axis=2)

    def to_scalar(self, probs: jax.Array):
        # probs: [batch, n, support]
        def f(probs):
            return jnp.sum(probs * self._centers)

        return jax.vmap(jax.vmap(f))(probs)

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        return jnp.argmax(
            self.to_scalar(self.get_q(params, convert_jax(obses), key)), axis=1, keepdims=True
        )

    def train_step(self, steps, gradient_steps):
        # Sample a batch from the replay buffer
        for _ in range(gradient_steps):
            self.train_steps_count += 1
            if self.prioritized_replay:
                data = self.replay_buffer.sample(self.batch_size, self.prioritized_replay_beta0)
            else:
                data = self.replay_buffer.sample(self.batch_size)

            (
                self.params,
                self.target_params,
                self.opt_state,
                loss,
                t_mean,
                new_priorities,
            ) = self._train_step(
                self.params,
                self.target_params,
                self.opt_state,
                self.train_steps_count,
                next(self.key_seq) if self.param_noise else None,
                **data
            )

            if self.prioritized_replay:
                self.replay_buffer.update_priorities(data["indexes"], new_priorities)

        if self.summary and steps % self.log_interval == 0:
            self.summary.add_scalar("loss/qloss", loss, steps)
            self.summary.add_scalar("loss/targets", t_mean, steps)

        return loss

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
        nxtobses,
        terminateds,
        weights=1,
        indexes=None,
    ):
        obses = convert_jax(obses)
        nxtobses = convert_jax(nxtobses)
        actions = jnp.expand_dims(actions.astype(jnp.int32), axis=2)
        not_terminateds = 1.0 - terminateds
        target_distribution = self._target(
            params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key
        )
        (loss, centropy), grad = jax.value_and_grad(self._loss, has_aux=True)(
            params, obses, actions, target_distribution, weights, key
        )
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        target_params = hard_update(params, target_params, steps, self.target_network_update_freq)
        new_priorities = None
        if self.prioritized_replay:
            new_priorities = centropy
        return (
            params,
            target_params,
            opt_state,
            loss,
            self.to_scalar(jnp.expand_dims(target_distribution, 1)).mean(),
            new_priorities,
        )

    def _loss(self, params, obses, actions, target_distribution, weights, key):
        distribution = jnp.squeeze(
            jnp.take_along_axis(self.get_q(params, obses, key), actions, axis=1)
        )
        cross_entropy = -jnp.sum(target_distribution * jnp.log(distribution + 1e-6), axis=1)
        return jnp.mean(cross_entropy), cross_entropy

    def _target(
        self, params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key
    ):
        next_prob = self.get_q(target_params, nxtobses, key)
        if self.double_q:
            next_action_q = self.to_scalar(self.get_q(params, nxtobses, key))
        else:
            next_action_q = self.to_scalar(next_prob)
        next_actions = jnp.expand_dims(jnp.argmax(next_action_q, axis=1), axis=(1, 2))
        next_distribution = jnp.squeeze(jnp.take_along_axis(next_prob, next_actions, axis=1))

        if self.munchausen:
            next_sub_q, tau_log_pi_next = q_log_pi(next_action_q, self.munchausen_entropy_tau)
            pi_next = jax.nn.softmax(next_sub_q / self.munchausen_entropy_tau)
            next_centers = self._centers - jnp.sum(pi_next * tau_log_pi_next, axis=1, keepdims=True)

            q_k_targets = self.to_scalar(self.get_q(target_params, obses, key))
            q_sub_targets, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau)
            log_pi = q_sub_targets - self.munchausen_entropy_tau * tau_log_pi
            munchausen_addon = jnp.take_along_axis(log_pi, jnp.squeeze(actions, axis=2), axis=1)

            rewards = rewards + self.munchausen_alpha * jnp.clip(
                munchausen_addon, a_min=-1, a_max=0
            )
        else:
            next_centers = self._centers
        target_centers = self._gamma * not_terminateds * next_centers + rewards  # [32, 51]

        target_distribution = self.to_probs(target_centers, next_distribution)  # [32, 51]
        return target_distribution

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=100,
        tb_log_name="C51",
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
