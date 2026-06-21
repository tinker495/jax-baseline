from copy import deepcopy
from itertools import repeat

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


class APE_X_QRDQN(Ape_X_Family):
    _run_name = "Ape_X_QRDQN"

    def __init__(
        self,
        workers,
        model_builder_maker,
        runtime,
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
            self.n_support,
            self.policy_kwargs,
        )

        self.preproc, self.model, self.params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)

        self.quantile = (
            jnp.linspace(0.0, 1.0, self.n_support + 1)[1:]
            + jnp.linspace(0.0, 1.0, self.n_support + 1)[:-1]
        ) / 2.0  # [support]
        self.quantile = jax.device_put(
            jnp.expand_dims(self.quantile, axis=(0, 1))
        )  # [1 x 1 x support]

        self.actor_builder = self.get_actor_builder()

        self.get_q = jax.jit(self.get_q)
        self._loss = jax.jit(self._loss)
        self._target = jax.jit(self._target)
        self._train_step = jax.jit(self._train_step)

    def get_q(self, params, obses, key=None) -> jnp.ndarray:
        return self.model(params, key, self.preproc(params, key, obses))

    def get_actor_builder(self):
        gamma = self._gamma
        action_size = self.action_size[0]
        param_noise = self.param_noise
        quantile = self.quantile
        delta = self.delta

        def builder():
            if param_noise:
                key_seq = key_gen(42)
            else:
                key_seq = repeat(None)

            def get_abs_td_error(
                model, preproc, params, obses, actions, rewards, nxtobses, terminateds, key
            ):
                q_values = jnp.take_along_axis(
                    model(params, key, preproc(params, key, convert_jax(obses))),
                    jnp.expand_dims(actions.astype(jnp.int32), axis=2),
                    axis=1,
                )
                next_q = model(params, key, preproc(params, key, convert_jax(nxtobses)))
                next_actions = jnp.expand_dims(
                    jnp.argmax(jnp.mean(next_q, axis=2), axis=1), axis=(1, 2)
                )
                next_vals = jnp.squeeze(
                    jnp.take_along_axis(next_q, next_actions, axis=1)
                )  # batch x support
                target = rewards + gamma * (1.0 - terminateds) * next_vals
                loss = QuantileHuberLosses(
                    jnp.expand_dims(target, axis=2), q_values, quantile, delta
                )
                return jnp.squeeze(loss)

            def actor(model, preproc, params, obses, key):
                q_values = model(params, key, preproc(params, key, convert_jax(obses)))
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
            self.params, self.target_params, self.opt_state, steps, next(self.key_seq), **data
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
        batch_idxes = jnp.arange(self.batch_size).reshape(-1, self.mini_batch_size)
        obses_batch = [o[batch_idxes] for o in obses]
        actions_batch = actions[batch_idxes]
        rewards_batch = rewards[batch_idxes]
        nxtobses_batch = [o[batch_idxes] for o in nxtobses]
        not_terminateds_batch = not_terminateds[batch_idxes]
        weights_batch = weights[batch_idxes]

        def f(carry, data):
            params, opt_state, key = carry
            obses, actions, rewards, nxtobses, not_terminateds, weights = data
            key, *subkeys = jax.random.split(key, 3)
            targets = self._target(
                params,
                target_params,
                obses,
                actions,
                rewards,
                nxtobses,
                not_terminateds,
                subkeys[0],
            )
            (loss, abs_error), grad = jax.value_and_grad(self._loss, has_aux=True)(
                params, obses, actions, targets, weights, subkeys[1]
            )
            updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state, key), (loss, targets, abs_error)

        (params, opt_state, key), (loss, targets, abs_error) = jax.lax.scan(
            f,
            (params, opt_state, key),
            (
                obses_batch,
                actions_batch,
                rewards_batch,
                nxtobses_batch,
                not_terminateds_batch,
                weights_batch,
            ),
        )
        target_params = hard_update(params, target_params, steps, self.target_network_update_freq)
        new_priorities = jnp.reshape(abs_error, (-1,))
        return params, target_params, opt_state, jnp.mean(loss), jnp.mean(targets), new_priorities

    def _loss(self, params, obses, actions, targets, weights, key):
        theta_loss_tile = jnp.take_along_axis(
            self.get_q(params, obses, key), actions, axis=1
        )  # batch x 1 x support
        logit_valid_tile = jnp.expand_dims(targets, axis=2)  # batch x support x 1
        loss = QuantileHuberLosses(logit_valid_tile, theta_loss_tile, self.quantile, self.delta)
        return (
            jnp.mean(loss * weights),
            loss,
        )  # remove weight multiply cpprb weight is something wrong

    def _target(
        self, params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key
    ):
        next_q = self.get_q(target_params, nxtobses, key)

        if self.munchausen:
            if self.double_q:
                next_q_mean = jnp.mean(self.get_q(params, nxtobses, key), axis=2)
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
                q_k_targets = jnp.mean(self.get_q(params, obses, key), axis=2)
            else:
                q_k_targets = jnp.mean(self.get_q(target_params, obses, key), axis=2)
            _, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau)
            munchausen_addon = jnp.take_along_axis(tau_log_pi, jnp.squeeze(actions, axis=2), axis=1)

            rewards = rewards + self.munchausen_alpha * jnp.clip(munchausen_addon, min=-1, max=0)
        else:
            if self.double_q:
                next_actions = jnp.expand_dims(
                    jnp.argmax(jnp.mean(self.get_q(params, nxtobses, key), axis=2), axis=1),
                    axis=(1, 2),
                )
            else:
                next_actions = jnp.expand_dims(
                    jnp.argmax(jnp.mean(next_q, axis=2), axis=1), axis=(1, 2)
                )
            next_vals = not_terminateds * jnp.squeeze(
                jnp.take_along_axis(next_q, next_actions, axis=1)
            )  # batch x support
        return (next_vals * self._gamma) + rewards  # batch x support
