import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax

from functools import partial
from haiku_baselines.DQN.base_class import Q_Network_Family
from haiku_baselines.DQN.network import Model
from haiku_baselines.common.Module import PreProcess
from haiku_baselines.common.utils import hard_update, convert_jax

class DQN(Q_Network_Family):
    def __init__(self, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.3,
                 exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, gradient_steps=1, batch_size=32, double_q=True,
                 dualing_model = False, n_step = 1, learning_starts=1000, target_network_update_freq=2000, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_eps=1e-6, 
                 param_noise=False, munchausen=False, log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None):
        
        super(DQN, self).__init__(env, gamma, learning_rate, buffer_size, exploration_fraction,
                 exploration_final_eps, exploration_initial_eps, train_freq, gradient_steps, batch_size, double_q,
                 dualing_model, n_step, learning_starts, target_network_update_freq, prioritized_replay,
                 prioritized_replay_alpha, prioritized_replay_beta0, prioritized_replay_eps, 
                 param_noise, munchausen, log_interval, tensorboard_log, _init_setup_model, policy_kwargs, 
                 full_tensorboard_log, seed)
        
        if _init_setup_model:
            self.setup_model() 
            
    def setup_model(self):
        self.key,sub_key = jax.random.split(self.key)
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
        if 'cnn_mode' in self.policy_kwargs.keys():
            cnn_mode = self.policy_kwargs['cnn_mode']
            del self.policy_kwargs['cnn_mode']
        self.preproc = hk.transform(lambda x: PreProcess(self.observation_space, cnn_mode=cnn_mode)(x))
        self.model = hk.transform(lambda x: Model(self.action_size,
                           dualing=self.dualing_model,noisy=self.param_noise,
                           **self.policy_kwargs)(x))
        pre_param = self.preproc.init(sub_key,
                            [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
        model_param = self.model.init(sub_key,
                            self.preproc.apply(pre_param, 
                            None, [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space]))
        self.params = hk.data_structures.merge(pre_param, model_param)
        self.target_params = self.params
        
        self.optimizer = optax.adamw(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        

        self.get_q = jax.jit(self.get_q)
        #self.learner_step = jax.jit(self.learner_step)
    
    #@jax.jit
    def get_q(self, params, obses) -> jnp.ndarray:
        feature = self.preproc.apply(params, None, obses)
        return self.model.apply(params, None, feature)
        
    def _get_actions(self, obses) -> np.ndarray:
        return np.asarray(jnp.argmax(self.get_q(self.params,convert_jax(obses)),axis=1))
    
    def train_step(self, steps):
        # Sample a batch from the replay buffer
        if self.prioritized_replay:
            data = self.replay_buffer.sample(self.batch_size,self.prioritized_replay_beta0)
        else:
            data = self.replay_buffer.sample(self.batch_size)
            
        self.params, self.target_params, self.opt_state, loss, t_mean = \
            self._train_step(self.params, self.target_params, self.opt_state, steps, **data)
            
        if self.summary and steps % self.log_interval == 0:
            self.summary.add_scalar("loss/qloss", loss, steps)
            self.summary.add_scalar("loss/targets", t_mean, steps)
        
    #@jax.jit
    def _loss(self, params, obses, actions, targets, weights=1):
        vals = self.get_q(params,obses)[actions]
        return jnp.mean(weights*jnp.square(vals - targets))
    
    #@jax.jit
    def _target(self,params,target_params, obses, actions, rewards, nxtobses, not_dones):
        next_q = self.get_q(target_params,nxtobses)
        if self.double_q:
            next_actions = jnp.argmax(self.get_q(params,nxtobses),axis=1)
        else:
            next_actions = jnp.argmax(next_q)
            
        if self.munchausen:
            logsum = jax.nn.logsumexp((next_q - jnp.max(next_q,axis=1,keepdims=True))/self.munchausen_entropy_tau, 1, keepdims=True)
            tau_log_pi_next = next_q - jnp.max(next_q,axis=1,keepdims=True) - self.munchausen_entropy_tau*logsum
            pi_target = jax.nn.softmax(next_q/self.munchausen_entropy_tau,dim=1)
            next_vals = jnp.sum(pi_target*not_dones*(next_q[next_actions] - tau_log_pi_next),keepdims=True)
            
            q_k_targets = self.get_q(target_params,obses)
            v_k_target = jnp.max(q_k_targets,axis=1,keepdims=True)
            logsum = jax.nn.logsumexp((q_k_targets - v_k_target)/self.munchausen_entropy_tau, 1, keepdims=True)
            log_pi = q_k_targets - v_k_target - self.munchausen_entropy_tau*logsum
            munchausen_addon = log_pi[actions]
            
            rewards += self.munchausen_alpha*jnp.clamp(munchausen_addon, min=-1, max=0)
        else:
            next_vals = not_dones * next_q[next_actions]
        return jax.lax.stop_gradient((next_vals * self._gamma) + rewards)
    

    def _train_step(self, params, target_params, opt_state, steps, obses, actions, rewards, nxtobses, dones, weights=1, indexes=None):
        obses = convert_jax(obses); nxtobses = convert_jax(nxtobses); actions = actions.astype('int'); not_dones = 1 - dones
        targets = self._target(params, target_params,obses, actions, rewards, nxtobses, not_dones)
        #jax.vmap(partial(self._target,params, target_params))(obses, actions, rewards, nxtobses, not_dones)
        loss,grad = jax.value_and_grad(self._loss)(params, obses, actions, targets, weights)
        updates, opt_state = self.optimizer.update(grad, opt_state, params)
        online_params = optax.apply_updates(params, updates)
        hard_update(online_params,target_params,steps,self.target_network_update_freq)
        return online_params, target_params, opt_state, loss, jnp.mean(targets)

    
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):
        super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)