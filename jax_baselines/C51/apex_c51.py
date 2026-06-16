from copy import deepcopy
from itertools import repeat

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.APE_X.base_class import Ape_X_Family
from jax_baselines.core.seeding import key_gen
from jax_baselines.math.distributional import (
    CategoricalBackend,
    MunchausenSpec,
    distributional_td_target,
)
from jax_baselines.math.jax_utils import convert_jax
from jax_baselines.math.param_updates import hard_update


class APE_X_C51(Ape_X_Family):
    def __init__(
        self,
        workers,
        model_builder_maker,
        manager=None,
        categorial_bar_n=51,
        categorial_max=250,
        categorial_min=-250,
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

        self.categorial_bar_n = categorial_bar_n
        self.categorial_max = categorial_max
        self.categorial_min = categorial_min

        super().__init__(
            workers,
            model_builder_maker,
            manager=manager,
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
            self.categorial_bar_n,
            self.policy_kwargs,
        )

        self.preproc, self.model, self.params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)

        self.categorial_bar = jnp.expand_dims(
            jnp.linspace(self.categorial_min, self.categorial_max, self.categorial_bar_n),
            axis=0,
        )  # [1, 51]
        self.delta_bar = jax.device_put(
            (self.categorial_max - self.categorial_min) / (self.categorial_bar_n - 1)
        )
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
        categorial_bar_n = self.categorial_bar_n
        categorial_min = self.categorial_min
        categorial_max = self.categorial_max
        categorial_bar = self.categorial_bar
        delta_bar = self.delta_bar

        def builder():
            if param_noise:
                key_seq = key_gen(42)
            else:
                key_seq = repeat(None)

            def get_abs_td_error(
                model, preproc, params, obses, actions, rewards, nxtobses, terminateds, key
            ):
                distribution = jnp.squeeze(
                    jnp.take_along_axis(
                        model(
                            params,
                            key,
                            preproc(params, key, convert_jax(obses)),
                        ),
                        jnp.expand_dims(actions.astype(jnp.int32), axis=2),
                        axis=1,
                    )
                )

                next_q = model(params, key, preproc(params, key, convert_jax(nxtobses)))
                next_actions = jnp.expand_dims(
                    jnp.argmax(jnp.sum(next_q * categorial_bar, axis=2), axis=1),
                    axis=(1, 2),
                )
                next_distribution = jnp.squeeze(jnp.take_along_axis(next_q, next_actions, axis=1))
                next_categorial = (1.0 - terminateds) * categorial_bar
                target_categorial = (next_categorial * gamma) + rewards

                Tz = jnp.clip(target_categorial, categorial_min, categorial_max)
                C51_B = ((Tz - categorial_min) / delta_bar).astype(jnp.float32)
                C51_L = jnp.floor(C51_B).astype(jnp.int32)
                C51_H = jnp.ceil(C51_B).astype(jnp.int32)
                C51_L = jnp.where(
                    (C51_H > 0) * (C51_L == C51_H), C51_L - 1, C51_L
                )  # C51_L.at[].add(-1)
                C51_H = jnp.where(
                    (C51_L < (categorial_bar_n - 1)) * (C51_L == C51_H),
                    C51_H + 1,
                    C51_H,
                )  # C51_H.at[].add(1)

                def tdist(next_distribution, C51_L, C51_H, C51_B):
                    exact = C51_L == C51_H
                    target_distribution = jnp.zeros((self.categorial_bar_n))

                    w_l = jnp.where(
                        exact,
                        next_distribution,
                        next_distribution * (C51_H.astype(jnp.float32) - C51_B),
                    )
                    w_u = jnp.where(
                        exact,
                        jnp.zeros_like(next_distribution),
                        next_distribution * (C51_B - C51_L.astype(jnp.float32)),
                    )

                    target_distribution = target_distribution.at[C51_L].add(w_l)
                    target_distribution = target_distribution.at[C51_H].add(w_u)
                    return target_distribution

                target_distribution = jax.vmap(tdist, in_axes=(0, 0, 0, 0))(
                    next_distribution, C51_L, C51_H, C51_B
                )
                loss = jnp.mean(target_distribution * (-jnp.log(distribution + 1e-5)), axis=1)
                return jnp.squeeze(loss)

            def actor(model, preproc, params, obses, key):
                q_values = jnp.sum(
                    model(params, key, preproc(params, key, convert_jax(obses))) * categorial_bar,
                    axis=2,
                )
                return jnp.argmax(q_values, axis=1)

            if param_noise:

                def get_action(actor, params, obs, epsilon, key):
                    return np.asarray(actor(params, obs, key))[0]

            else:

                def get_action(actor, params, obs, epsilon, key):
                    if epsilon <= np.random.uniform(0, 1):
                        actions = np.asarray(actor(params, obs, key))[0]
                    else:
                        actions = np.random.choice(action_size)
                    return actions

            def random_action(params, obs, epsilon, key):
                return np.random.choice(action_size)

            return get_abs_td_error, actor, get_action, random_action, key_seq

        return builder

    def train_step(self, steps, gradient_steps):
        for _ in range(gradient_steps):
            self.train_steps_count += 1
            data = self.replay_buffer.sample(self.batch_size, self.prioritized_replay_beta0)

            (
                self.params,
                self.target_params,
                self.opt_state,
                loss,
                t_mean,
                new_priorities,
            ) = self._train_step(
                self.params, self.target_params, self.opt_state, steps, next(self.key_seq), **data
            )

            self.replay_buffer.update_priorities(data["indexes"], new_priorities)

        if steps % self.log_interval == 0:
            log_dict = {"loss/qloss": float(loss), "loss/targets": float(t_mean)}
            self.logger_server.log_trainer.remote(steps, log_dict)

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
            target_distribution = self._target(
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
                params, obses, actions, target_distribution, weights, subkeys[1]
            )
            updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state, key), (loss, target_distribution, abs_error)

        (params, opt_state, key), (loss, target_distribution, abs_error) = jax.lax.scan(
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
        return (
            params,
            target_params,
            opt_state,
            jnp.mean(loss),
            jnp.mean(
                jnp.sum(
                    jnp.reshape(target_distribution, (-1, self.categorial_bar_n))
                    * self.categorial_bar,
                    axis=1,
                )
            ),
            new_priorities,
        )

    def _loss(self, params, obses, actions, target_distribution, weights, key):
        distribution = jnp.squeeze(
            jnp.take_along_axis(self.get_q(params, obses, key), actions, axis=1)
        )
        loss = jnp.mean(target_distribution * (-jnp.log(distribution + 1e-5)), axis=1)
        return jnp.mean(loss * weights), loss

    def _target(
        self, params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key
    ):
        next_dists = self.get_q(target_params, nxtobses, key)
        online_next_dists = self.get_q(params, nxtobses, key) if self.double_q else None

        backend = CategoricalBackend(
            support=self.categorial_bar[0],
            support_min=self.categorial_min,
            support_max=self.categorial_max,
            delta=self.delta_bar,
            n_bins=self.categorial_bar_n,
        )

        if self.munchausen:
            behavior_dists = self.get_q(params if self.double_q else target_params, obses, key)
            munchausen = MunchausenSpec(
                alpha=self.munchausen_alpha, tau=self.munchausen_entropy_tau
            )
        else:
            behavior_dists = None
            munchausen = None

        return distributional_td_target(
            next_dists=next_dists,
            actions=jnp.squeeze(actions, axis=2),
            reward=rewards,
            not_terminated=not_terminateds,
            gamma=self._gamma,
            backend=backend,
            online_next_dists=online_next_dists,
            behavior_dists=behavior_dists,
            munchausen=munchausen,
        )

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        run_name="Ape_X_C51",
        reset_num_timesteps=True,
        replay_wrapper=None,
        logger_factory=None,
        progress_factory=None,
    ):
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            run_name,
            reset_num_timesteps,
            replay_wrapper,
            logger_factory=logger_factory,
            progress_factory=progress_factory,
        )
