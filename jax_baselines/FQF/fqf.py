from copy import deepcopy

import jax
import jax.numpy as jnp
import optax

from jax_baselines.common.losses import FQFQuantileLosses, QuantileHuberLosses
from jax_baselines.common.utils import convert_jax, hard_update
from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.FQF.network.haiku import model_builder_maker


class FQF(Q_Network_Family):
    def __init__(
        self,
        env,
        gamma=0.995,
        learning_rate=3e-4,
        buffer_size=100000,
        exploration_fraction=0.3,
        n_support=32,
        delta=1.0,
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
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-3,
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
        super().__init__(
            env,
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

        self.name = "FQF"
        self.n_support = n_support
        self.delta = delta
        self.ent_coef = 1e-2

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        self.model_bulder = model_builder_maker(
            self.observation_space,
            self.action_size,
            self.dueling_model,
            self.param_noise,
            self.n_support,
            self.policy_kwargs,
        )

        self.preproc, self.fpf, self.model, self.params = self.model_bulder(
            next(self.key_seq), print_model=True
        )
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)

        self.tile_n = self.n_support

        self.get_q = jax.jit(self.get_q)
        self._get_actions = jax.jit(self._get_actions)
        self._loss = jax.jit(self._loss)
        self._target = jax.jit(self._target)
        self._train_step = jax.jit(self._train_step)

    def get_q(self, params, feature, tau, key=None) -> jnp.ndarray:
        return self.model(params, key, feature, tau)

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        feature = self.preproc(params, key, convert_jax(obses))
        _, tau, _ = self.fpf(params, key, feature)
        return jnp.expand_dims(
            jnp.argmax(jnp.mean(self.get_q(params, feature, tau, key), axis=2), axis=1),
            axis=1,
        )

    def train_step(self, steps, gradient_steps):
        for _ in range(gradient_steps):
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
                steps,
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
        dones,
        weights=1,
        indexes=None,
    ):
        obses = convert_jax(obses)
        nxtobses = convert_jax(nxtobses)
        actions = jnp.expand_dims(actions.astype(jnp.int32), axis=2)
        not_dones = 1.0 - dones
        targets = self._target(
            params, target_params, obses, actions, rewards, nxtobses, not_dones, key
        )
        (loss, abs_error), grad = jax.value_and_grad(self._loss, has_aux=True)(
            params, obses, actions, targets, weights, key
        )
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        target_params = hard_update(params, target_params, steps, self.target_network_update_freq)
        new_priorities = None
        if self.prioritized_replay:
            new_priorities = abs_error
        return params, target_params, opt_state, loss, jnp.mean(targets), new_priorities

    def _loss(self, params, obses, actions, targets, weights, key):
        feature = self.preproc(params, key, obses)
        tau, tau_hats, entropy = self.fpf(params, key, feature)
        theta_loss_tile = jnp.take_along_axis(
            self.get_q(params, feature, jax.lax.stop_gradient(tau_hats), key),
            actions,
            axis=1,
        )  # batch x 1 x support
        logit_valid_tile = jnp.expand_dims(targets, axis=2)  # batch x support x 1
        hubber = QuantileHuberLosses(
            theta_loss_tile,
            logit_valid_tile,
            jax.lax.stop_gradient(jnp.expand_dims(tau_hats, axis=1)),
            self.delta,
        )
        q_loss = jnp.mean(
            hubber * weights
        )  # remove weight multiply cpprb weight is something wrong
        tau_vals = jax.lax.stop_gradient(
            jnp.take_along_axis(self.get_q(params, feature, tau[:, 1:-1], key), actions, axis=1)
        )  # batch x 1 x support
        quantile_loss = jnp.mean(
            FQFQuantileLosses(
                jnp.squeeze(tau_vals),
                jax.lax.stop_gradient(jnp.squeeze(theta_loss_tile)),
                tau,
            )
        )
        entropy_loss = self.ent_coef * jnp.mean(entropy)
        total_loss = q_loss + quantile_loss + entropy_loss
        return total_loss, hubber

    def _target(self, params, target_params, obses, actions, rewards, nxtobses, not_dones, key):
        feature = self.preproc(target_params, key, nxtobses)
        _, tau_hats, _ = self.fpf(target_params, key, feature)
        next_q = self.get_q(
            target_params,
            feature,
            tau_hats,
            key,
        )

        if self.munchausen:
            next_q_mean = jnp.mean(next_q, axis=2)  # [batch x action]
            logsum = jax.nn.logsumexp(
                (next_q_mean - jnp.max(next_q_mean, axis=1, keepdims=True))
                / self.munchausen_entropy_tau,
                axis=1,
                keepdims=True,
            )  # []
            tau_log_pi_next = jnp.expand_dims(
                next_q_mean
                - jnp.max(next_q_mean, axis=1, keepdims=True)
                - self.munchausen_entropy_tau * logsum,
                axis=2,
            )
            pi_next = jnp.expand_dims(
                jax.nn.softmax(next_q_mean / self.munchausen_entropy_tau, axis=1),
                axis=2,
            )
            next_vals = (
                jnp.sum(
                    pi_next * (next_q - jnp.expand_dims(tau_log_pi_next, axis=2)),
                    axis=1,
                )
                * not_dones
            )

            q_k_targets = jnp.mean(
                self.get_q(
                    target_params,
                    self.preproc(target_params, key, obses),
                    tau_hats,
                    key,
                ),
                axis=2,
            )
            v_k_target = jnp.max(q_k_targets, axis=1, keepdims=True)
            logsum = jax.nn.logsumexp(
                (q_k_targets - v_k_target) / self.munchausen_entropy_tau,
                axis=1,
                keepdims=True,
            )
            log_pi = jnp.expand_dims(
                q_k_targets - v_k_target - self.munchausen_entropy_tau * logsum, axis=2
            )
            munchausen_addon = jnp.take_along_axis(log_pi, actions, axis=1)

            rewards += self.munchausen_alpha * jnp.clip(munchausen_addon, a_min=-1, a_max=0)
        else:
            if self.double_q:
                feature = self.preproc(params, key, nxtobses)
                next_actions = jnp.expand_dims(
                    jnp.argmax(
                        jnp.mean(self.get_q(params, feature, nxtobses, tau_hats, key), axis=2),
                        axis=1,
                    ),
                    axis=(1, 2),
                )
            else:
                next_actions = jnp.expand_dims(
                    jnp.argmax(jnp.mean(next_q, axis=2), axis=1), axis=(1, 2)
                )
            next_vals = not_dones * jnp.squeeze(
                jnp.take_along_axis(next_q, next_actions, axis=1)
            )  # batch x support
        return (next_vals * self._gamma) + rewards  # batch x support

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=100,
        tb_log_name="FQF",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        tb_log_name = tb_log_name + "({:d})".format(self.n_support)
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            tb_log_name,
            reset_num_timesteps,
            replay_wrapper,
        )
