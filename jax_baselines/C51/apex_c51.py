from copy import deepcopy
from itertools import repeat

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.APE_X.base_class import Ape_X_Family
from jax_baselines.common.utils import convert_jax, hard_update, key_gen, q_log_pi


class APE_X_C51(Ape_X_Family):
    def __init__(
        self,
        workers,
        model_builder_maker,
        manager=None,
        gamma=0.995,
        learning_rate=5e-5,
        buffer_size=50000,
        exploration_initial_eps=0.8,
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
        log_interval=10,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        categorial_bar_n=51,
        categorial_max=10,
        categorial_min=-10,
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
        compress_memory=False,
    ):
        super().__init__(
            workers,
            model_builder_maker,
            manager,
            gamma,
            learning_rate,
            buffer_size,
            exploration_initial_eps,
            exploration_decay,
            batch_num,
            mini_batch_size,
            double_q,
            dueling_model,
            n_step,
            learning_starts,
            target_network_update_freq,
            gradient_steps,
            prioritized_replay_alpha,
            prioritized_replay_beta0,
            prioritized_replay_eps,
            param_noise,
            munchausen,
            log_interval,
            log_dir,
            _init_setup_model,
            policy_kwargs,
            full_tensorboard_log,
            seed,
            optimizer,
            compress_memory,
        )

        self.categorial_bar_n = categorial_bar_n
        self.categorial_max = categorial_max
        self.categorial_min = categorial_min

        if _init_setup_model:
            self.setup_model()

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
        self._categorial_bar = jnp.expand_dims(self.categorial_bar, axis=0)  # [1, 1, 51]
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
        action_size = self.action_size[0]
        param_noise = self.param_noise
        categorial_bar = np.asarray(self._categorial_bar)

        def builder():
            if param_noise:
                key_seq = key_gen(42)
            else:
                # make repeat None
                key_seq = repeat(None)

            def actor(model, preproc, params, obses, key):
                q_values = jnp.sum(
                    model(params, key, preproc(params, key, convert_jax(obses))) * categorial_bar,
                    axis=2,
                )
                return jnp.argmax(q_values, axis=1)

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

            return actor, get_action, key_seq

        return builder

    def train_step(self, steps, gradient_steps):
        # Sample a batch from the replay buffer
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
                self.params,
                self.target_params,
                self.opt_state,
                self.train_steps_count,
                next(self.key_seq),
                **data
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
                    target_distribution * self._categorial_bar,
                    axis=2,
                )
            ),
            new_priorities,
        )

    def _loss(self, params, obses, actions, target_distribution, weights, key):
        distribution = jnp.squeeze(
            jnp.take_along_axis(self.get_q(params, obses, key), actions, axis=1)
        )
        cross_entropy = -jnp.sum(target_distribution * jnp.log(distribution + 1e-6), axis=1)
        return jnp.mean(cross_entropy * weights), cross_entropy

    def _target(
        self, params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key
    ):
        next_distributions = self.get_q(target_params, nxtobses, key)
        if self.double_q:
            next_action_q = jnp.sum(
                self.get_q(params, nxtobses, key) * self._categorial_bar, axis=2
            )
        else:
            next_action_q = jnp.sum(next_distributions * self._categorial_bar, axis=2)

        def tdist(next_distribution, target_categorial):
            Tz = jnp.clip(
                target_categorial, self.categorial_min, self.categorial_max
            )  # clip to range of bar
            C51_B = ((Tz - self.categorial_min) / self.delta_bar).astype(
                jnp.float32
            )  # bar index as float
            C51_L = jnp.floor(C51_B).astype(jnp.int32)  # bar lower index as int
            C51_H = jnp.ceil(C51_B).astype(jnp.int32)  # bar higher index as int
            C51_L = jnp.where(
                (C51_L > 0) * (C51_L == C51_H), C51_L - 1, C51_L
            )  # C51_L.at[].add(-1)
            C51_H = jnp.where(
                (C51_H < (self.categorial_bar_n - 1)) * (C51_L == C51_H), C51_H + 1, C51_H
            )  # C51_H.at[].add(1)

            def tdist(next_distribution, C51_L, C51_H, C51_b):
                target_distribution = jnp.zeros((self.categorial_bar_n))
                target_distribution = target_distribution.at[C51_L].add(
                    next_distribution * (C51_H.astype(jnp.float32) - C51_b)
                )
                target_distribution = target_distribution.at[C51_H].add(
                    next_distribution * (C51_b - C51_L.astype(jnp.float32))
                )
                return target_distribution

            return jax.vmap(tdist, in_axes=(0, 0, 0, 0))(next_distribution, C51_L, C51_H, C51_B)

        if self.munchausen:
            next_sub_q, tau_log_pi_next = q_log_pi(next_action_q, self.munchausen_entropy_tau)
            pi_next = jax.nn.softmax(next_sub_q / self.munchausen_entropy_tau)  # [32, action_size]
            next_categorials = self._categorial_bar - jnp.expand_dims(
                tau_log_pi_next, axis=2
            )  # [32, action_size, 51]

            if self.double_q:
                q_k_targets = jnp.sum(self.get_q(params, obses, key) * self._categorial_bar, axis=2)
            else:
                q_k_targets = jnp.sum(
                    self.get_q(target_params, obses, key) * self._categorial_bar, axis=2
                )
            _, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau)
            munchausen_addon = jnp.take_along_axis(tau_log_pi, jnp.squeeze(actions, axis=2), axis=1)

            rewards = rewards + self.munchausen_alpha * jnp.clip(
                munchausen_addon, a_min=-1, a_max=0
            )  # [32, 1]
            target_categorials = jnp.expand_dims(
                self._gamma * not_terminateds, axis=2
            ) * next_categorials + jnp.expand_dims(
                rewards, axis=2
            )  # [32, action_size, 51]
            target_distributions = jax.vmap(tdist, in_axes=(1, 1), out_axes=1)(
                next_distributions, target_categorials
            )
            target_distribution = jnp.sum(
                jnp.expand_dims(pi_next, axis=2) * target_distributions, axis=1
            )
        else:
            next_actions = jnp.expand_dims(jnp.argmax(next_action_q, axis=1), axis=(1, 2))
            next_distribution = jnp.squeeze(
                jnp.take_along_axis(next_distributions, next_actions, axis=1)
            )
            target_categorial = (
                self._gamma * not_terminateds * self.categorial_bar + rewards
            )  # [32, 51]
            target_distribution = tdist(next_distribution, target_categorial)

        return target_distribution

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="APE_X",
        run_name="Ape_X_C51",
    ):
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            experiment_name,
            run_name,
        )
