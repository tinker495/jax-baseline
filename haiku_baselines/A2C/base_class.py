import gymnasium as gym
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from tqdm.auto import trange
from collections import deque

from haiku_baselines.common.base_classes import TensorboardWriter, save, restore, select_optimizer
from haiku_baselines.common.buffers import EpochBuffer
from haiku_baselines.common.utils import convert_states
from haiku_baselines.common.worker import gymMultiworker

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from gym import spaces

class Actor_Critic_Policy_Gradient_Family(object):
    def __init__(self, env, gamma=0.995,learning_rate=3e-4, batch_size=32, val_coef=0.2, ent_coef=0.01,
                 log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None, optimizer = 'adamw'):
        
        self.env = env
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        self.key_seq = hk.PRNGSequence(self.seed)
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.val_coef = val_coef
        self.ent_coef = ent_coef
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        
        self.params = None
        self.save_path = None
        self.optimizer = select_optimizer(optimizer,self.learning_rate)
        
        self.get_env_setup()
        
    def save_params(self, path):
        save(path, self.params)
            
    def load_params(self, path):
        self.params = restore(path)
        
    def get_memory_setup(self):
        self.buffer = EpochBuffer(self.batch_size, self.observation_space, self.worker_size, [1] if self.action_type == 'discrete' else self.action_size)
        
    def get_env_setup(self):
        print("----------------------env------------------------")
        if isinstance(self.env,UnityEnvironment):
            print("unity-ml agent environmet")
            self.env.reset()
            group_name = list(self.env.behavior_specs.keys())[0]
            group_spec = self.env.behavior_specs[group_name]
            self.env.step()
            dec, term = self.env.get_steps(group_name)
            self.group_name = group_name
            
            self.observation_space = [list(spec.shape) for spec in group_spec.observation_specs]
            if group_spec.action_spec.continuous_size == 0:
                self.action_size = [branch for branch in group_spec.action_spec.discrete_branches]
                self.action_type = 'discrete'
                self.conv_action = lambda a: ActionTuple(discrete=a)
            else:
                self.action_size = [group_spec.action_spec.continuous_size]
                self.action_type = 'continuous'
                self.conv_action = lambda a: ActionTuple(continuous=np.clip(a, -3.0, 3.0) / 3.0) #np.clip(a, -3.0, 3.0) / 3.0) 
            self.worker_size = len(dec.agent_id)
            self.env_type = "unity"
            
        elif isinstance(self.env,gym.Env) or isinstance(self.env,gym.Wrapper):
            print("openai gym environmet")
            action_space = self.env.action_space
            observation_space = self.env.observation_space
            self.observation_space = [list(observation_space.shape)]
            if not isinstance(action_space, spaces.Box):
                self.action_size = [action_space.n]
                self.action_type = 'discrete'
                self.conv_action = lambda a: a[0][0]
            else:
                self.action_size = [action_space.shape[0]]
                self.action_type = 'continuous'
                self.conv_action = lambda a: a[0]
            self.worker_size = 1
            self.env_type = "gym"
            
        elif isinstance(self.env,gymMultiworker):
            print("gymMultiworker")
            env_info = self.env.env_info
            self.observation_space = [list(env_info['observation_space'].shape)]
            if not isinstance(env_info['action_space'], spaces.Box):
                self.action_size = [env_info['action_space'].n]
                self.action_type = 'discrete'
            else:
                self.action_size = [env_info['action_space'].shape[0]]
                self.action_type = 'continuous'
            self.worker_size = self.env.worker_num
            self.env_type = "gymMultiworker"
    
        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", self.worker_size)
        print("-------------------------------------------------")
        if self.action_type == 'discrete':
            self._get_actions = self._get_actions_discrete
            self.get_logprob = self.get_logprob_discrete
            self._loss = self._loss_discrete
            self.actions = self.action_discrete
        elif self.action_type == 'continuous':
            self._get_actions = self._get_actions_continuous
            self.get_logprob = self.get_logprob_continuous
            self._loss = self._loss_continuous
            self.actions = self.action_continuous
    
    def setup_model(self):
        pass
    
    def _train_step(self, steps):
        pass
    
    def _get_actions_discrete(self):
        pass
    
    def _get_actions_continuous(self):
        pass
    
    def _loss_discrete(self):
        pass
    
    def _get_actions(self, params, obses) -> np.ndarray:
        pass
    
    def action_discrete(self,obs,steps):
        pass
    
    def action_continuous(self,obs,steps):
        pass
    
    def _loss_continuous(self):
        pass
    
    def get_logprob_discrete(self,params, feature, action, key, out_prob=False):
        pass
    
    def get_logprob_continuous(self,params, feature, action, key, out_prob=False):
        pass
        
    def learn(self, total_timesteps, callback=None, log_interval=1000, tb_log_name="Q_network",
              reset_num_timesteps=True, replay_wrapper=None):
        
        self.buffer.clear()
        pbar = trange(total_timesteps, miniters=log_interval, smoothing=0.01)
        with TensorboardWriter(self.tensorboard_log, tb_log_name) as (self.summary, self.save_path):
            if self.env_type == "unity":
                self.learn_unity(pbar, callback, log_interval)
            if self.env_type == "gym":
                self.learn_gym(pbar, callback, log_interval)
            if self.env_type == "gymMultiworker":
                self.learn_gymMultiworker(pbar, callback, log_interval)
            self.save_params(self.save_path)
    
    def discription(self):
        return "score : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque),np.mean(self.lossque)
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
            actions = self.actions(obses)
            action_tuple = self.conv_action(actions)

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
                    term_obses = [np.concatenate((to,o),axis=0) for to,o in zip(term_obses,newterm_obs)]
                    term_rewards += list(term.reward)
                    term_done += list(term.interrupted)
            nxtobs = convert_states(dec.obs)
            done = np.full((self.worker_size),False)
            terminal = np.full((self.worker_size),False)
            reward = dec.reward
            term_on = len(term_ids) > 0
            if term_on:
                term_ids = np.asarray(term_ids)
                term_rewards = np.asarray(term_rewards)
                term_done = np.asarray(term_done)
                for n,t in zip(nxtobs,term_obses):
                    n.at[term_ids].set(t)
                done[term_ids] = ~term_done
                terminal[term_ids] = True
                reward[term_ids] = term_rewards
            self.scores += reward
            self.buffer.add(obses, actions, np.expand_dims(reward,axis=1), nxtobs, np.expand_dims(done,axis=1), np.expand_dims(terminal,axis=1))
            obses = nxtobs
            if term_on:
                if self.summary:
                    self.summary.add_scalar("env/episode_reward", np.mean(self.scores[term_ids]), steps)
                    self.summary.add_scalar("env/episode len",np.mean(self.eplen[term_ids]),steps)
                    self.summary.add_scalar("env/time over",np.mean(1 - done[term_ids].astype(np.float32)),steps)
                self.scoreque.extend(self.scores[term_ids])
                self.scores[term_ids] = 0
                self.eplen[term_ids] = 0
            
            if (steps + 1) % self.batch_size == 0: #train in step the environments
                loss = self.train_step(steps)
                self.lossque.append(loss)
                self.buffer.clear()
            
            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description(self.discription())
        
    def learn_gym(self, pbar, callback=None, log_interval=100):
        state, info = self.env.reset()
        state = [np.expand_dims(state,axis=0)]
        self.scores = np.zeros([self.worker_size])
        self.eplen = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        self.lossque = deque(maxlen=10)
        for steps in pbar:
            self.eplen += 1
            actions = self.actions(state)
            next_state, reward, terminal, truncated, info = self.env.step(self.conv_action(actions))
            next_state = [np.expand_dims(next_state,axis=0)]
            self.buffer.add(state, actions[0], reward, next_state, terminal, truncated)
            self.scores[0] += reward
            state = next_state
            if terminal or truncated:
                self.scoreque.append(self.scores[0])
                if self.summary:
                    self.summary.add_scalar("env/episode_reward", self.scores[0], steps)
                    self.summary.add_scalar("env/episode len",self.eplen[0],steps)
                    self.summary.add_scalar("env/time over",float(truncated),steps)
                self.scores[0] = 0
                self.eplen[0] = 0
                state, info = self.env.reset()
                state = [np.expand_dims(state,axis=0)]
                
            if (steps + 1) % self.batch_size == 0: #train in step the environments
                loss = self.train_step(steps)
                self.lossque.append(loss)
                self.buffer.clear()
            
            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description(self.discription())
                
    def learn_gymMultiworker(self, pbar, callback=None, log_interval=100):
        state,_,_,_,_,_ = self.env.get_steps()
        self.scores = np.zeros([self.worker_size])
        self.eplen = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        self.lossque = deque(maxlen=10)
        for steps in pbar:
            self.eplen += 1
            actions = self.actions([state])
            self.env.step(actions)
            
            next_states,rewards,dones,terminals,end_states,end_idx = self.env.get_steps()
            nxtstates = np.copy(next_states)
            if end_states is not None:
                nxtstates[end_idx] = end_states
                if self.summary:
                    self.summary.add_scalar("env/episode_reward", np.mean(self.scores[end_idx]), steps)
                    self.summary.add_scalar("env/episode len",np.mean(self.eplen[end_idx]),steps)
                    self.summary.add_scalar("env/time over",np.mean(1 - dones[end_idx].astype(np.float32)),steps)
                self.scoreque.extend(self.scores[end_idx])
                self.scores[end_idx] = 0
                self.eplen[end_idx] = 0
            self.buffer.add([state], actions, np.expand_dims(rewards,axis=1), [nxtstates], np.expand_dims(dones,axis=1), np.expand_dims(terminals,axis=1))
            self.scores += rewards
            state = next_states
            
            if (steps + 1) % self.batch_size == 0: #train in step the environments
                loss = self.train_step(steps)
                self.lossque.append(loss)
                self.buffer.clear()
            
            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description(self.discription())
    
    def test(self, episode = 10, tb_log_name=None):
        if tb_log_name is None:
            tb_log_name = self.save_path
        
        directory = tb_log_name
        if self.env_type == "gym":
            self.test_gym(episode, directory)
        if self.env_type == "gymMultiworker":
            self.test_gymMultiworker(episode, directory)
    
    def test_unity(self, episode,directory):
        pass
    
    def test_gymMultiworker(self, episode,directory):
        from gymnasium.wrappers import RecordVideo
        env_id = self.env.env_id
        from haiku_baselines.common.atari_wrappers import make_wrap_atari,get_env_type
        env_type, env_id = get_env_type(env_id)
        if  env_type == 'atari_env':
            env = make_wrap_atari(env_id,clip_rewards=True)
        else:
            env = gym.make(env_id, render_mode='rgb_array')
        Render_env = RecordVideo(env, directory)
        for i in range(episode):
            state, info = Render_env.reset()
            state = [np.expand_dims(state,axis=0)]
            terminal = False
            truncated = False
            episode_rew = 0
            while not (terminal or truncated):
                actions = self.actions(state)
                observation, reward, terminal, truncated, info = Render_env.step(actions[0][0] if self.action_type == 'discrete' else actions[0])
                state = [np.expand_dims(observation,axis=0)]
                episode_rew += reward
            print("episod reward :", episode_rew)
    
    def test_gym(self, episode,directory):
        from gymnasium.wrappers import RecordVideo
        Render_env = RecordVideo(self.env, directory, episode_trigger = lambda x: True)
        for i in range(episode):
            state, info = Render_env.reset()
            state = [np.expand_dims(state,axis=0)]
            terminal = False
            truncated = False
            episode_rew = 0
            while not (terminal or truncated):
                actions = self.actions(state)
                observation, reward, terminal, truncated, info = Render_env.step(actions[0][0] if self.action_type == 'discrete' else actions[0])
                state = [np.expand_dims(observation,axis=0)]
                episode_rew += reward
            print("episod reward :", episode_rew)