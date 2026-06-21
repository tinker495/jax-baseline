import random
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.APE_X.base_class import Ape_X_Family
from jax_baselines.core.seeding import key_gen
from jax_baselines.math.jax_utils import convert_jax
from jax_baselines.math.losses import QuantileHuberLosses
from jax_baselines.math.param_updates import hard_update
from jax_baselines.math.policy_math import q_log_pi


class APE_X_IQN(Ape_X_Family):
    _run_name = "Ape_X_IQN"

    def __init__(
        self,
        workers,
        model_builder_maker,
        runtime,
        CVaR=1.0,
        n_support=200,
        delta=1.0,
        gamma=0.995,
        learning_rate=5e-5,
        buffer_size=50000,
        exploration_initial_eps=0.9,
        exploration_decay=0.7,
        batch_num=16,
        mini_batch_size=512,
        double_q=False,
        dueling_model=False,
        n_step=1,
        learning_starts=1000,
        target_network_update_freq=2000,
        gradient_steps=1,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-3,
        param_noise=False,
        munchausen=False,
        log_interval=200,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        seed=None,
        optimizer_factory=None,
        compress_memory=False,
        multi_replay_factory=None,
        worker_replay_factory=None,
    ):

        self.n_support = n_support
        self.delta = delta
        self.CVaR = CVaR
        self.risk_avoid = CVaR != 1.0

        super().__init__(
            workers,
            model_builder_maker,
            runtime=runtime,
            gamma=gamma,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            exploration_initial_eps=exploration_initial_eps,
            exploration_decay=exploration_decay,
            batch_num=batch_num,
            mini_batch_size=mini_batch_size,
            double_q=double_q,
            dueling_model=dueling_model,
            n_step=n_step,
            learning_starts=learning_starts,
            target_network_update_freq=target_network_update_freq,
            gradient_steps=gradient_steps,
            prioritized_replay_alpha=prioritized_replay_alpha,
            prioritized_replay_beta0=prioritized_replay_beta0,
            prioritized_replay_eps=prioritized_replay_eps,
            param_noise=param_noise,
            munchausen=munchausen,
            log_interval=log_interval,
            log_dir=log_dir,
            _init_setup_model=_init_setup_model,
            policy_kwargs=policy_kwargs,
            seed=seed,
            optimizer_factory=optimizer_factory,
            compress_memory=compress_memory,
            multi_replay_factory=multi_replay_factory,
            worker_replay_factory=worker_replay_factory,
        )

    def setup_model(self):
        self.model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.dueling_model,
            self.param_noise,
            self.policy_kwargs,
        )

        self.preproc, self.model, self.params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)
        self.actor_builder = self.get_actor_builder()

        self.get_q = jax.jit(self.get_q)
        self._loss = jax.jit(self._loss)
        self._target = jax.jit(self._target)
        self._train_step = jax.jit(self._train_step)

    def get_q(self, params, obses, tau, key=None) -> jnp.ndarray:
        return self.model(params, key, self.preproc(params, key, obses), tau)

    def get_actor_builder(self):
        gamma = self._gamma
        action_size = self.action_size[0]
        param_noise = self.param_noise
        delta = self.delta
        n_support = self.n_support
        CVaR = self.CVaR

        def builder():
            key_seq = key_gen(random.randint(0, 1000000))

            def get_abs_td_error(
                model, preproc, params, obses, actions, rewards, nxtobses, terminateds, key
            ):
                key1, key2, key3 = jax.random.split(key, 3)
                conv_obses = convert_jax(obses)
                batch_size = conv_obses[0].shape[0]
                tau = jax.random.uniform(key1, (batch_size, n_support))
                next_tau = jax.random.uniform(key2, (batch_size, n_support))
                q_values = jnp.take_along_axis(
                    model(params, key3, preproc(params, key3, conv_obses), tau),
                    jnp.expand_dims(actions.astype(jnp.int32), axis=2),
                    axis=1,
                )
                next_q = model(params, key3, preproc(params, key3, convert_jax(nxtobses)), next_tau)
                next_actions = jnp.expand_dims(
                    jnp.argmax(jnp.mean(next_q, axis=2), axis=1), axis=(1, 2)
                )
                next_vals = jnp.squeeze(
                    jnp.take_along_axis(next_q, next_actions, axis=1)
                )  # batch x support
                target = rewards + gamma * (1.0 - terminateds) * next_vals
                loss = QuantileHuberLosses(jnp.expand_dims(target, axis=2), q_values, tau, delta)
                return jnp.squeeze(loss)

            def actor(model, preproc, params, obses, key):
                # CVaR-distorted quantile sampling for risk-averse acting,
                # mirroring local IQN._get_actions; the priority/TD-error path
                # (get_abs_td_error) deliberately uses plain U[0,1] like iqn.py _loss/_target.
                tau = jax.random.uniform(key, (1, n_support)) * CVaR
                q_values = model(params, key, preproc(params, key, convert_jax(obses)), tau)
                return jnp.expand_dims(jnp.argmax(jnp.mean(q_values, axis=2), axis=1), axis=1)

            if param_noise:

                def get_action(actor, params, obs, epsilon, key):
                    return int(np.asarray(actor(params, obs, key))[0])

            else:

                def get_action(actor, params, obs, epsilon, key):
                    if epsilon <= np.random.uniform(0, 1):
                        actions = int(np.asarray(actor(params, obs, key))[0])
                    else:
                        actions = np.random.choice(action_size)
                    return actions

            def random_action(params, obs, epsilon, key):
                return np.random.choice(action_size)

            return get_abs_td_error, actor, get_action, random_action, key_seq

        return builder

    def _invoke_train_step(self, steps, data):
        return self._train_step(
            self.params,
            self.target_params,
            self.opt_state,
            steps,
            next(self.key_seq) if self.param_noise else None,
            **data,
        )

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
        key1, key2 = jax.random.split(key, 2)
        targets = self._target(
            params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key1
        )
        (loss, abs_error), grad = jax.value_and_grad(self._loss, has_aux=True)(
            params, obses, actions, targets, weights, key2
        )
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        target_params = hard_update(params, target_params, steps, self.target_network_update_freq)
        new_priorities = abs_error
        return params, target_params, opt_state, loss, jnp.mean(targets), new_priorities

    def _loss(self, params, obses, actions, targets, weights, key):
        tau = jax.random.uniform(key, (self.batch_size, self.n_support))
        theta_loss_tile = jnp.take_along_axis(
            self.get_q(params, obses, tau, key), actions, axis=1
        )  # batch x 1 x support
        logit_valid_tile = jnp.expand_dims(targets, axis=2)  # batch x support x 1
        loss = QuantileHuberLosses(
            logit_valid_tile, theta_loss_tile, jnp.expand_dims(tau, axis=1), self.delta
        )
        return jnp.mean(loss * weights), loss

    def _target(
        self, params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key
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
            )
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

            rewards = rewards + self.munchausen_alpha * jnp.clip(munchausen_addon, min=-1, max=0)
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
        return (next_vals * self._gamma) + rewards  # batch x support
