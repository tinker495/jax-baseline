from copy import deepcopy
from itertools import repeat

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.APE_X.base_class import Ape_X_Family
from jax_baselines.common.utils import convert_jax, hard_update, key_gen, q_log_pi


class APE_X_DQN(Ape_X_Family):
    def __init__(self, workers, model_builder_maker, **kwargs):
        # No subclass-specific fields to pre-set; keep pattern consistent
        super().__init__(workers, model_builder_maker, **kwargs)

    def setup_model(self):
        self.model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.dueling_model,
            self.param_noise,
            self.policy_kwargs,
        )
        self.actor_builder = self.get_actor_builder()

        self.preproc, self.model, self.params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.target_params = deepcopy(self.params)

        self.opt_state = self.optimizer.init(self.params)

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
        prioritized_replay_eps = self.prioritized_replay_eps

        def builder():
            if param_noise:
                key_seq = key_gen(42)
            else:
                # make repeat None
                key_seq = repeat(None)

            def get_abs_td_error(
                model, preproc, params, obses, actions, rewards, nxtobses, terminateds, key
            ):
                q_values = jnp.take_along_axis(
                    model(params, key, preproc(params, key, convert_jax(obses))),
                    actions.astype(jnp.int32),
                    axis=1,
                )
                next_q_values = jnp.max(
                    model(params, key, preproc(params, key, convert_jax(nxtobses))),
                    axis=1,
                    keepdims=True,
                )
                target = rewards + gamma * (1.0 - terminateds) * next_q_values
                td_error = q_values - target
                return jnp.squeeze(jnp.abs(td_error)) + prioritized_replay_eps

            def actor(model, preproc, params, obses, key):
                q_values = model(params, key, preproc(params, key, convert_jax(obses)))
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

            def random_action(params, obs, epsilon, key):
                return np.random.choice(action_size)

            return get_abs_td_error, actor, get_action, random_action, key_seq

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
        actions = actions.astype(jnp.int32)
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
        vals = jnp.take_along_axis(self.get_q(params, obses, key), actions, axis=1)
        error = jnp.squeeze(vals - targets)
        loss = jnp.square(error)
        # loss = hubberloss(error, delta=1.0)
        return jnp.mean(loss * jnp.squeeze(weights)), jnp.abs(
            error
        )  # remove weight multiply cpprb weight is something wrong

    def _target(
        self, params, target_params, obses, actions, rewards, nxtobses, not_terminateds, key
    ):
        next_q = self.get_q(target_params, nxtobses, key)

        if self.munchausen:
            if self.double_q:
                next_sub_q, tau_log_pi_next = q_log_pi(
                    self.get_q(params, nxtobses, key), self.munchausen_entropy_tau
                )
            else:
                next_sub_q, tau_log_pi_next = q_log_pi(next_q, self.munchausen_entropy_tau)
            pi_next = jax.nn.softmax(next_sub_q / self.munchausen_entropy_tau)
            next_vals = (
                jnp.sum(pi_next * (next_q - tau_log_pi_next), axis=1, keepdims=True)
                * not_terminateds
            )

            q_k_targets = self.get_q(target_params, obses, key)
            q_sub_targets, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau)
            log_pi = q_sub_targets - self.munchausen_entropy_tau * tau_log_pi
            munchausen_addon = jnp.take_along_axis(log_pi, actions, axis=1)

            rewards = rewards + self.munchausen_alpha * jnp.clip(
                munchausen_addon, a_min=-1, a_max=0
            )
        else:
            if self.double_q:
                next_actions = jnp.argmax(self.get_q(params, nxtobses, key), axis=1, keepdims=True)
            else:
                next_actions = jnp.argmax(next_q, axis=1, keepdims=True)
            next_vals = not_terminateds * jnp.take_along_axis(next_q, next_actions, axis=1)
        return (next_vals * self._gamma) + rewards

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        run_name="Ape_X_DQN",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            run_name,
            reset_num_timesteps,
            replay_wrapper,
        )
