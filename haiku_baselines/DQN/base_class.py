import gymnasium as gym
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from tqdm.auto import trange
from collections import deque

from haiku_baselines.common.base_classes import TensorboardWriter, save, restore, select_optimizer
#from haiku_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer, EpisodicReplayBuffer, PrioritizedEpisodicReplayBuffer
from haiku_baselines.common.cpprb_buffers import ReplayBuffer, NstepReplayBuffer, PrioritizedReplayBuffer, PrioritizedNstepReplayBuffer
from haiku_baselines.common.schedules import LinearSchedule, ConstantSchedule
from haiku_baselines.common.utils import convert_states
from haiku_baselines.common.worker import gymMultiworker

from mlagents_envs.environment import UnityEnvironment, ActionTuple

class Q_Network_Family(object):
    def __init__(self, env, gamma=0.995, learning_rate=5e-5, buffer_size=50000, exploration_fraction=0.3,
                 exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, gradient_steps=1, batch_size=32, double_q=False,
                 dueling_model = False, n_step = 1, learning_starts=1000, target_network_update_freq=2000, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_eps=1e-3, 
                 param_noise=False, munchausen=False, log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None, optimizer = 'adamw', compress_memory = False):
        
        self.env = env
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        self.key_seq = hk.PRNGSequence(self.seed)
        
        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_freq = int(np.ceil(target_network_update_freq / train_freq) * train_freq)
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self._gamma = np.power(gamma,n_step) #n_step gamma
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.double_q = double_q
        self.dueling_model = dueling_model
        self.n_step_method = (n_step > 1)
        self.n_step = n_step
        self.munchausen = munchausen
        self.munchausen_alpha = 0.9
        self.munchausen_entropy_tau = 0.03
        
        self.params = None
        self.target_params = None
        self.save_path = None
        self.optimizer = select_optimizer(optimizer,self.learning_rate,1e-2/self.batch_size)

        self.compress_memory = compress_memory
        
        self.get_env_setup()
        self.get_memory_setup()
        
    def save_params(self, path):
        save(path, self.params)
            
    def load_params(self, path):
        self.params = self.target_params = restore(path)
        
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
            self.action_size = [branch for branch in group_spec.action_spec.discrete_branches]
            self.worker_size = len(dec.agent_id)
            self.env_type = "unity"
            
        elif isinstance(self.env,gym.Env) or isinstance(self.env,gym.Wrapper):
            print("openai gym environmet")
            action_space = self.env.action_space
            observation_space = self.env.observation_space
            self.observation_space = [list(observation_space.shape)]
            self.action_size = [action_space.n]
            self.worker_size = 1
            self.env_type = "gym"
            
        elif isinstance(self.env,gymMultiworker):
            print("gymMultiworker")
            env_info = self.env.env_info
            self.observation_space = [list(env_info['observation_space'].shape)]
            self.action_size = [env_info['action_space'].n]
            self.worker_size = self.env.worker_num
            self.env_type = "gymMultiworker"
    
        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", self.worker_size)
        print("-------------------------------------------------")
        
    def get_memory_setup(self):
        '''
        if self.prioritized_replay:
            if self.n_step_method:
                self.replay_buffer = PrioritizedEpisodicReplayBuffer(self.buffer_size,self.observation_space, self.worker_size, 1,
                                                                     self.n_step, self.gamma, self.prioritized_replay_alpha)
            else:
                self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size,self.observation_space, self.worker_size, 1,
                                                             self.prioritized_replay_alpha)

        else:
            if self.n_step_method:
                self.replay_buffer = EpisodicReplayBuffer(self.buffer_size,self.observation_space, self.worker_size, 1, self.n_step, self.gamma)
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size,self.observation_space, self.worker_size, 1)
        '''
        if self.prioritized_replay:
            if self.n_step_method:
                self.replay_buffer = PrioritizedNstepReplayBuffer(self.buffer_size,self.observation_space, 1, self.worker_size,
                                                                     self.n_step, self.gamma, self.prioritized_replay_alpha)
            else:
                self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size,self.observation_space, 1)

        else:
            if self.n_step_method:
                self.replay_buffer = NstepReplayBuffer(self.buffer_size,self.observation_space, 1, self.worker_size, self.n_step, self.gamma)
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size,self.observation_space, 1)
    
    def setup_model(self):
        pass
    
    def _train_step(self, steps):
        pass
    
    def _get_actions(self, params, obses) -> np.ndarray:
        pass
    
    def actions(self,obs,epsilon):
        if epsilon <= np.random.uniform(0,1):
            actions = np.asarray(self._get_actions(self.params,obs,next(self.key_seq) if self.param_noise else None))
        else:
            actions = np.random.choice(self.action_size[0], [self.worker_size,1])
        return actions

    def discription(self):
        if self.param_noise:
            return "score : {:.3f}, loss : {:.3f} |".format(
                                        np.mean(self.scoreque),np.mean(self.lossque)
                                        )
        else:
            return "score : {:.3f}, epsilon : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque),self.update_eps,np.mean(self.lossque)
                                    )
        
    def learn(self, total_timesteps, callback=None, log_interval=1000, tb_log_name="Q_network",
              reset_num_timesteps=True, replay_wrapper=None):
        if self.munchausen:
            tb_log_name = "M-" + tb_log_name
        if self.param_noise:
            tb_log_name = "Noisy_" + tb_log_name
        if self.dueling_model:
            tb_log_name = "Dueling_" + tb_log_name
        if self.double_q:
            tb_log_name = "Double_" + tb_log_name
        if self.n_step_method:
            tb_log_name = "{}Step_".format(self.n_step) + tb_log_name
        if self.prioritized_replay:
            tb_log_name = tb_log_name + "+PER"
        
        if self.param_noise:
            self.exploration = ConstantSchedule(0)
        else:
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                                    initial_p=self.exploration_initial_eps,
                                                    final_p=self.exploration_final_eps)
        self.update_eps = 1.0
            
        pbar = trange(total_timesteps, miniters=log_interval)
        with TensorboardWriter(self.tensorboard_log, tb_log_name) as (self.summary, self.save_path):
            if self.env_type == "unity":
                self.learn_unity(pbar, callback, log_interval)
            if self.env_type == "gym":
                self.learn_gym(pbar, callback, log_interval)
            if self.env_type == "gymMultiworker":
                self.learn_gymMultiworker(pbar, callback, log_interval)
            self.save_params(self.save_path)
    
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
            actions = self.actions(obses,self.update_eps)
            action_tuple = ActionTuple(discrete=actions)
            old_obses = obses

            self.env.set_actions(self.group_name, action_tuple)
            self.env.step()
            
            if steps > self.learning_starts and steps % self.train_freq == 0: #train in step the environments
                self.update_eps = self.exploration.value(steps)
                loss = self.train_step(steps,self.gradient_steps)
                self.lossque.append(loss)
            
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
            obses = convert_states(dec.obs)
            nxtobs = [np.copy(o) for o in obses]
            done = np.full((self.worker_size),False)
            terminal = np.full((self.worker_size),False)
            reward = dec.reward
            term_on = len(term_ids) > 0
            if term_on:
                term_ids = np.asarray(term_ids)
                term_rewards = np.asarray(term_rewards)
                term_done = np.asarray(term_done)
                for n,t in zip(nxtobs,term_obses):
                    n[term_ids] = t
                done[term_ids] = ~term_done
                terminal[term_ids] = True
                reward[term_ids] = term_rewards
            self.scores += reward
            self.replay_buffer.add(old_obses, actions, reward, nxtobs, done, terminal)
            if term_on:
                if self.summary:
                    self.summary.add_scalar("env/episode_reward", np.mean(self.scores[term_ids]), steps)
                    self.summary.add_scalar("env/episode len",np.mean(self.eplen[term_ids]),steps)
                    self.summary.add_scalar("env/time over",np.mean(1 - done[term_ids].astype(np.float32)),steps)
                self.scoreque.extend(self.scores[term_ids])
                self.scores[term_ids] = 0
                self.eplen[term_ids] = 0
            
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
            actions = self.actions(state,self.update_eps)
            next_state, reward, terminal, truncated, info = self.env.step(actions[0][0])
            next_state = [np.expand_dims(next_state,axis=0)]
            self.replay_buffer.add(state, actions[0], reward, next_state, terminal, truncated)
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
                
            if steps > self.learning_starts and steps % self.train_freq == 0:
                self.update_eps = self.exploration.value(steps)
                loss = self.train_step(steps,self.gradient_steps)
                self.lossque.append(loss)
            
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
            self.update_eps = self.exploration.value(steps)
            actions = self.actions([state],self.update_eps)
            self.env.step(actions)

            if steps > self.learning_starts and steps % self.train_freq == 0:
                self.update_eps = self.exploration.value(steps)
                loss = self.train_step(steps,self.gradient_steps)
                self.lossque.append(loss)
            
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
            self.replay_buffer.add([state], actions, rewards, [nxtstates], dones, terminals)
            self.scores += rewards
            state = next_states
            
            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description(self.discription())
    
    def test(self, episode = 10, tb_log_name=None):
        if tb_log_name is None:
            tb_log_name = self.save_path
        
        directory = tb_log_name
        if self.env_type == "gym":
            self.test_gym(episode, directory)
    
    def test_unity(self, episode,directory):
        pass
    
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
                actions = self.actions(state,0.001)
                observation, reward, terminal, truncated, info = Render_env.step(actions[0][0])
                state = [np.expand_dims(observation,axis=0)]
                episode_rew += reward
            print("episod reward :", episode_rew)