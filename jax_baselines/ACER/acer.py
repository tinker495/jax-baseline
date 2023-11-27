import gymnasium as gym
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from copy import deepcopy
from copy import deepcopy

from collections import deque

from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.A2C.network import Actor, Critic
from jax_baselines.model.haiku.Module import PreProcess
from jax_baselines.common.utils import convert_states
from jax_baselines.common.utils import convert_jax, discount_with_terminal

from jax_baselines.common.cpprb_buffers import ReplayBuffer, PrioritizedReplayBuffer


class ACER(Actor_Critic_Policy_Gradient_Family):
    def __init__(
        self,
        env,
        gamma=0.995,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=32,
        val_coef=0.2,
        ent_coef=0.5,
        log_interval=200,
        tensorboard_log=None,
        _init_setup_model=True,
        policy_kwargs=None,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-3,
        n_step=1,
        full_tensorboard_log=False,
        seed=None,
        optimizer="rmsprop",
    ):
        super().__init__(
            env,
            gamma,
            learning_rate,
            batch_size,
            val_coef,
            ent_coef,
            log_interval,
            tensorboard_log,
            _init_setup_model,
            policy_kwargs,
            full_tensorboard_log,
            seed,
            optimizer,
        )

        self.buffer_size = buffer_size
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self._gamma = self.gamma**n_step
        self.n_step_method = n_step > 1
        self.n_step = n_step

        if _init_setup_model:
            self.setup_model()

    def get_memory_setup(self):
        if not self.prioritized_replay:
            self.replay_buffer = ReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.worker_size,
                1 if self.action_type == "discrete" else self.action_size,
                self.n_step,
                self.gamma,
            )
        else:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.prioritized_replay_alpha,
                self.worker_size,
                1 if self.action_type == "discrete" else self.action_size,
                self.n_step,
                self.gamma,
            )

    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
        if "cnn_mode" in self.policy_kwargs.keys():
            cnn_mode = self.policy_kwargs["cnn_mode"]
            del self.policy_kwargs["cnn_mode"]
        self.preproc = hk.transform(
            lambda x: PreProcess(self.observation_space, cnn_mode=cnn_mode)(x)
        )
        self.actor = hk.transform(
            lambda x: Actor(self.action_size, self.action_type, **self.policy_kwargs)(x)
        )
        self.critic = hk.transform(lambda x: Critic(**self.policy_kwargs)(x))
        pre_param = self.preproc.init(
            next(self.key_seq),
            [np.zeros((1, *o), dtype=np.float32) for o in self.observation_space],
        )
        feature = self.preproc.apply(
            pre_param,
            None,
            [np.zeros((1, *o), dtype=np.float32) for o in self.observation_space],
        )
        actor_param = self.actor.init(next(self.key_seq), feature)
        critic_param = self.critic.init(next(self.key_seq), feature)
        self.params = hk.data_structures.merge(pre_param, actor_param, critic_param)

        self.opt_state = self.optimizer.init(self.params)

        print("----------------------model----------------------")
        print(jax.tree_map(lambda x: x.shape, pre_param))
        print(jax.tree_map(lambda x: x.shape, actor_param))
        print(jax.tree_map(lambda x: x.shape, critic_param))
        print("-------------------------------------------------")

        self._get_actions = jax.jit(self._get_actions)
        self._train_step = jax.jit(self._train_step)

    def _get_actions_discrete(self, params, obses, key=None) -> jnp.ndarray:
        prob = jax.nn.softmax(
            self.actor.apply(params, key, self.preproc.apply(params, key, convert_jax(obses))),
            axis=1,
        )
        return prob

    def _get_actions_continuous(self, params, obses, key=None) -> jnp.ndarray:
        mu, std = self.actor.apply(params, key, self.preproc.apply(params, key, convert_jax(obses)))
        return mu, jnp.exp(std)

    def get_logprob_discrete(self, prob, action, key, out_prob=False):
        prob = jnp.clip(jax.nn.softmax(prob), 1e-5, 1.0)
        action = action.astype(jnp.int32)
        if out_prob:
            return prob, jnp.log(jnp.take_along_axis(prob, action, axis=1))
        else:
            return jnp.log(jnp.take_along_axis(prob, action, axis=1))

    def get_logprob_continuous(self, prob, action, key, out_prob=False):
        mu, log_std = prob
        std = jnp.exp(log_std)
        if out_prob:
            return prob, -(
                0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-6)), axis=-1, keepdims=True)
                + jnp.sum(log_std, axis=-1, keepdims=True)
                + 0.5 * jnp.log(2 * np.pi) * jnp.asarray(action.shape[-1], dtype=jnp.float32)
            )
        else:
            return -(
                0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-6)), axis=-1, keepdims=True)
                + jnp.sum(log_std, axis=-1, keepdims=True)
                + 0.5 * jnp.log(2 * np.pi) * jnp.asarray(action.shape[-1], dtype=jnp.float32)
            )

    def discription(self):
        return "score : {:.3f}, loss : {:.3f} |".format(
            np.mean(self.scoreque), np.mean(self.lossque)
        )

    def action_discrete(self, obs, steps):
        prob = self._get_actions(self.params, obs)
        return prob, np.expand_dims(
            np.stack([np.random.choice(self.action_size[0], p=p) for p in prob], axis=0),
            axis=1,
        )

    def action_continuous(self, obs, steps):
        mu, std = self._get_actions(self.params, obs)
        return (mu, std), np.random.normal(mu, std)

    def train_step(self, steps):
        # Sample a batch from the replay buffer
        data = self.buffer.get_buffer()

        self.params, self.opt_state, critic_loss, actor_loss = self._train_step(
            self.params, self.opt_state, None, self.ent_coef, **data
        )

        if self.summary:
            self.summary.add_scalar("loss/critic_loss", critic_loss, steps)
            self.summary.add_scalar("loss/actor_loss", actor_loss, steps)

        return critic_loss

    def _train_step(
        self,
        params,
        opt_state,
        key,
        ent_coef,
        obses,
        actions,
        rewards,
        nxtobses,
        dones,
        terminals,
    ):
        obses = [convert_jax(o) for o in obses]
        nxtobses = [convert_jax(n) for n in nxtobses]
        value = [self.critic.apply(params, key, self.preproc.apply(params, key, o)) for o in obses]
        next_value = [
            self.critic.apply(params, key, self.preproc.apply(params, key, n)) for n in nxtobses
        ]
        targets = [
            discount_with_terminal(r, d, t, nv, self.gamma)
            for r, d, t, nv in zip(rewards, dones, terminals, next_value)
        ]
        obses = [jnp.vstack(list(zo)) for zo in zip(*obses)]
        actions = jnp.vstack(actions)
        value = jnp.vstack(value)
        targets = jnp.vstack(targets)
        adv = targets - value
        (total_loss, (critic_loss, actor_loss)), grad = jax.value_and_grad(
            self._loss, has_aux=True
        )(params, obses, actions, targets, adv, ent_coef, key)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, critic_loss, actor_loss

    def _loss_discrete(self, params, obses, actions, targets, adv, ent_coef, key):
        feature = self.preproc.apply(params, key, obses)
        vals = self.critic.apply(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))

        prob, log_prob = self.get_logprob(
            self.actor.apply(params, key, feature), actions, key, out_prob=True
        )
        actor_loss = -jnp.mean(log_prob * adv)
        entropy = prob * jnp.log(prob)
        entropy_loss = jnp.mean(entropy)
        total_loss = self.val_coef * critic_loss + actor_loss - ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss)

    def _loss_continuous(self, params, obses, actions, targets, adv, ent_coef, key):
        feature = self.preproc.apply(params, key, obses)
        vals = self.critic.apply(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))

        prob, log_prob = self.get_logprob(
            self.actor.apply(params, key, feature), actions, key, out_prob=True
        )
        actor_loss = -jnp.mean(log_prob * adv)
        mu, log_std = prob
        entropy_loss = jnp.mean(jnp.abs(mu) + jnp.abs(log_std))
        total_loss = self.val_coef * critic_loss + actor_loss - ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss)

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=100,
        tb_log_name="ACER",
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

    def learn_unity(self, pbar, callback=None, log_interval=100):
        self.env.reset()
        self.env.step()
        dec, term = self.env.get_steps(self.group_name)
        self.scores = np.zeros([self.worker_size])
        self.eplen = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        self.lossque = deque(maxlen=10)
        obses = convert_states(dec.obs)
        for steps in pbar:
            self.eplen += 1
            prob, actions = self.actions(obses, steps)
            action_tuple = self.conv_action_tuple(actions)
            old_obses = obses

            self.env.set_actions(self.group_name, action_tuple)
            self.env.step()

            dec, term = self.env.get_steps(self.group_name)
            term_ids = list(term.agent_id)
            term_obses = convert_states(term.obs)
            term_rewards = list(term.reward)
            term_done = list(term.interrupted)
            while len(dec) == 0:
                self.env.step()
                dec, term = self.env.get_steps(self.group_name)
                if len(term.agent_id) > 0:
                    term_ids += list(term.agent_id)
                    newterm_obs = convert_states(term.obs)
                    term_obses = [
                        np.concatenate((to, o), axis=0) for to, o in zip(term_obses, newterm_obs)
                    ]
                    term_rewards += list(term.reward)
                    term_done += list(term.interrupted)
            obses = convert_states(dec.obs)
            nxtobs = [np.copy(o) for o in obses]
            done = np.full((self.worker_size), False)
            terminal = np.full((self.worker_size), False)
            reward = dec.reward
            term_on = len(term_ids) > 0
            if term_on:
                term_ids = np.asarray(term_ids)
                term_rewards = np.asarray(term_rewards)
                term_done = np.asarray(term_done)
                for n, t in zip(nxtobs, term_obses):
                    n[term_ids] = t
                done[term_ids] = ~term_done
                terminal[term_ids] = True
                reward[term_ids] = term_rewards
            self.scores += reward
            self.replay_buffer.add(
                old_obses,
                actions,
                np.expand_dims(reward, axis=1),
                nxtobs,
                np.expand_dims(done, axis=1),
                np.expand_dims(terminal, axis=1),
            )
            if term_on:
                if self.summary:
                    self.summary.add_scalar(
                        "env/episode_reward", np.mean(self.scores[term_ids]), steps
                    )
                    self.summary.add_scalar("env/episode len", np.mean(self.eplen[term_ids]), steps)
                    self.summary.add_scalar(
                        "env/time over",
                        np.mean(1 - done[term_ids].astype(np.float32)),
                        steps,
                    )
                self.scoreque.extend(self.scores[term_ids])
                self.scores[term_ids] = 0
                self.eplen[term_ids] = 0

            if (steps + 1) % self.batch_size == 0:  # train in step the environments
                loss = self.train_step(steps)
                self.lossque.append(loss)

            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description(self.discription())

    def learn_gym(self, pbar, callback=None, log_interval=100):
        state = [np.expand_dims(self.env.reset(), axis=0)]
        self.scores = np.zeros([self.worker_size])
        self.eplen = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        self.lossque = deque(maxlen=10)
        for steps in pbar:
            self.eplen += 1
            prob, actions = self.actions(state, steps)
            next_state, reward, terminal, truncated, info = self.env.step(
                actions[0][0] if self.action_type == "discrete" else actions[0]
            )
            next_state = [np.expand_dims(next_state, axis=0)]
            done = terminal
            if "TimeLimit.truncated" in info:
                done = not info["TimeLimit.truncated"]
            self.replay_buffer.add(state, actions[0], reward, next_state, done, terminal)
            self.scores[0] += reward
            state = next_state
            if terminal:
                self.scoreque.append(self.scores[0])
                if self.summary:
                    self.summary.add_scalar("env/episode_reward", self.scores[0], steps)
                    self.summary.add_scalar("env/episode len", self.eplen[0], steps)
                    self.summary.add_scalar("env/time over", float(not done), steps)
                self.scores[0] = 0
                self.eplen[0] = 0
                state = [np.expand_dims(self.env.reset(), axis=0)]

            if (steps + 1) % self.batch_size == 0:  # train in step the environments
                loss = self.train_step(steps)
                self.lossque.append(loss)

            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description(self.discription())

    def learn_gymMultiworker(self, pbar, callback=None, log_interval=100):
        state, _, _, _, _, _ = self.env.get_steps()
        self.scores = np.zeros([self.worker_size])
        self.eplen = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        self.lossque = deque(maxlen=10)
        for steps in pbar:
            self.eplen += 1
            prob, actions = self.actions([state], steps)
            self.env.step(actions)

            (
                next_states,
                rewards,
                dones,
                terminals,
                end_states,
                end_idx,
            ) = self.env.get_steps()
            nxtstates = np.copy(next_states)
            if end_states is not None:
                nxtstates[end_idx] = end_states
                if self.summary:
                    self.summary.add_scalar(
                        "env/episode_reward", np.mean(self.scores[end_idx]), steps
                    )
                    self.summary.add_scalar("env/episode len", np.mean(self.eplen[end_idx]), steps)
                    self.summary.add_scalar(
                        "env/time over",
                        np.mean(1 - dones[end_idx].astype(np.float32)),
                        steps,
                    )
                self.scoreque.extend(self.scores[end_idx])
                self.scores[end_idx] = 0
                self.eplen[end_idx] = 0
            self.replay_buffer.add(
                [state],
                actions,
                np.expand_dims(rewards, axis=1),
                [nxtstates],
                np.expand_dims(dones, axis=1),
                np.expand_dims(terminals, axis=1),
            )
            self.scores += rewards
            state = next_states

            if (steps + 1) % self.batch_size == 0:  # train in step the environments
                loss = self.train_step(steps)
                self.lossque.append(loss)

            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description(self.discription())

    def test_action(self, state):
        return self.actions(state, 0)

    def test(self, episode=10, tb_log_name=None):
        if tb_log_name is None:
            tb_log_name = self.save_path

        directory = tb_log_name
        if self.env_type == "gym":
            self.test_gym(episode, directory)
        if self.env_type == "gymMultiworker":
            self.test_gymMultiworker(episode, directory)

    def test_unity(self, episode, directory):
        pass

    def test_gymMultiworker(self, episode, directory):
        from colabgymrender.recorder import Recorder

        env_id = self.env.env_id
        from jax_baselines.common.atari_wrappers import make_wrap_atari, get_env_type

        env_type, env_id = get_env_type(env_id)
        if env_type == "atari":
            env = make_wrap_atari(env_id)
        else:
            env = gym.make(env_id)
        Render_env = Recorder(env, directory)
        for i in range(episode):
            state = [np.expand_dims(Render_env.reset(), axis=0)]
            terminal = False
            episode_rew = 0
            while not terminal:
                actions = self.test_action(state)
                observation, reward, terminal, info = Render_env.step(
                    actions[0][0] if self.action_type == "discrete" else actions[0]
                )
                state = [np.expand_dims(observation, axis=0)]
                episode_rew += reward
            print("episod reward :", episode_rew)

    def test_gym(self, episode, directory):
        from colabgymrender.recorder import Recorder

        Render_env = Recorder(self.env, directory)
        for i in range(episode):
            state = [np.expand_dims(Render_env.reset(), axis=0)]
            terminal = False
            episode_rew = 0
            while not terminal:
                actions = self.test_action(state)
                observation, reward, terminal, info = Render_env.step(
                    actions[0][0] if self.action_type == "discrete" else actions[0]
                )
                state = [np.expand_dims(observation, axis=0)]
                episode_rew += reward
            print("episod reward :", episode_rew)
