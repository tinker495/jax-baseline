import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax

from haiku_baselines.DQN.base_class import Q_Network_Family
from haiku_baselines.C51.network import Model
from haiku_baselines.common.Module import PreProcess
from haiku_baselines.common.utils import hard_update, convert_jax

class C51(Q_Network_Family):
    def __init__(self, env, gamma=0.99, learning_rate=5e-4, buffer_size=100000, exploration_fraction=0.3, categorial_bar_n = 51,
                 exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, gradient_steps=1, batch_size=32, double_q=True,
                 dualing_model = False, n_step = 1, learning_starts=1000, target_network_update_freq=2000, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_eps=1e-6, 
                 param_noise=False, munchausen=False, log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 categorial_max = 250, categorial_min = -250,
                 full_tensorboard_log=False, seed=None):
        
        super(C51, self).__init__(env, gamma, learning_rate, buffer_size, exploration_fraction,
                 exploration_final_eps, exploration_initial_eps, train_freq, gradient_steps, batch_size, double_q,
                 dualing_model, n_step, learning_starts, target_network_update_freq, prioritized_replay,
                 prioritized_replay_alpha, prioritized_replay_beta0, prioritized_replay_eps, 
                 param_noise, munchausen, log_interval, tensorboard_log, _init_setup_model, policy_kwargs, 
                 full_tensorboard_log, seed)
        
        self.categorial_bar_n = categorial_bar_n
        self.categorial_max = categorial_max
        self.categorial_min = categorial_min
        
        if _init_setup_model:
            self.setup_model()
            
    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
        if 'cnn_mode' in self.policy_kwargs.keys():
            cnn_mode = self.policy_kwargs['cnn_mode']
            del self.policy_kwargs['cnn_mode']
        self.preproc = hk.transform(lambda x: PreProcess(self.observation_space, cnn_mode=cnn_mode)(x))
        self.model = hk.transform(lambda x: Model(self.action_size,
                           dualing=self.dualing_model,noisy=self.param_noise,
                           **self.policy_kwargs)(x))
        pre_param = self.preproc.init(hk.next_rng_key(),
                            [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
        model_param = self.model.init(hk.next_rng_key(),
                            self.preproc.apply(pre_param, 
                            None, [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space]))
        self.params = hk.data_structures.merge(pre_param, model_param)
        self.target_params = self.params
        
        self.optimizer = optax.adamw(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        
        self.categorial_bar = jnp.expand_dims(jnp.linspace(self.categorial_min, self.categorial_max, self.categorial_bar_n),axis=0)
        self._categorial_bar = jnp.expand_dims(self.categorial_bar,axis=0)
        self.delta_bar = jax.device_put((self.categorial_max - self.categorial_min)/(self.categorial_bar_n - 1))
        
        offset = jnp.expand_dims(jnp.linspace(0, (self.batch_size - 1) * self.categorial_bar_n, self.batch_size),axis=-1)
        self.offset = jnp.broadcast_to(offset,(self.batch_size, self.categorial_bar_n)).astype(jnp.int32)
        
        print("----------------------model----------------------")
        print(jax.tree_map(lambda x: x.shape, pre_param))
        print(jax.tree_map(lambda x: x.shape, model_param))
        print("loss : logistic_distribution_loss")
        print("-------------------------------------------------")

        self.get_q = jax.jit(self.get_q)
        self._get_actions = jax.jit(self._get_actions)
        self._loss = jax.jit(self._loss)
        self._target = jax.jit(self._target)
        self._train_step = jax.jit(self._train_step)
    
    def get_q(self, params, obses, key = None) -> jnp.ndarray:
        return self.model.apply(params, key, self.preproc.apply(params, key, obses))
        
    def _get_actions(self, params, obses, key = None) -> jnp.ndarray:
        return jnp.expand_dims(jnp.argmax(
               jnp.sum(self.get_q(params,convert_jax(obses),key)*self._categorial_bar,axis=2)
               ,axis=1),axis=1)
    
    def train_step(self, steps, gradient_steps):
        # Sample a batch from the replay buffer
        for _ in range(gradient_steps):
            if self.prioritized_replay:
                data = self.replay_buffer.sample(self.batch_size,self.prioritized_replay_beta0)
            else:
                data = self.replay_buffer.sample(self.batch_size)
            
            self.params, self.target_params, self.opt_state, loss, t_mean, new_priorities = \
                self._train_step(self.params, self.target_params, self.opt_state, steps, 
                                 hk.next_rng_key() if self.param_noise else None,**data)
            
            if self.prioritized_replay:
                self.replay_buffer.update_priorities(data['indexes'], new_priorities)
            
        if self.summary and steps % self.log_interval == 0:
            self.summary.add_scalar("loss/qloss", loss, steps)
            self.summary.add_scalar("loss/targets", t_mean, steps)
            
        return loss

    def _train_step(self, params, target_params, opt_state, steps, key, 
                    obses, actions, rewards, nxtobses, dones, weights=1, indexes=None):
        obses = convert_jax(obses); nxtobses = convert_jax(nxtobses); actions = jnp.expand_dims(actions.astype(jnp.int32),axis=2); not_dones = 1.0 - dones
        target_distribution = self._target(params, target_params, obses, actions, rewards, nxtobses, not_dones, key)
        (loss,abs_error), grad = jax.value_and_grad(self._loss,has_aux = True)(params, obses, actions, target_distribution, weights, key)
        updates, opt_state = self.optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        target_params = hard_update(params, target_params, steps, self.target_network_update_freq)
        new_priorities = None
        if self.prioritized_replay:
            new_priorities = abs_error + self.prioritized_replay_eps
        return params, target_params, opt_state, loss, jnp.mean(jnp.sum(target_distribution*self.categorial_bar,axis=1)), new_priorities
    
    def _loss(self, params, obses, actions, target_distribution, weights, key):
        distribution = jnp.clip(
                        jnp.squeeze(jnp.take_along_axis(self.get_q(params, obses, key), actions, axis=1))
                        ,1e-5,1.0)
        loss = jnp.sum(-target_distribution * jnp.log(distribution),axis=1)
        return jnp.mean(loss* weights), loss
    
    def _target(self,params, target_params, obses, actions, rewards, nxtobses, not_dones, key):
        next_q = self.get_q(target_params,nxtobses,key)
        if self.double_q:
            next_actions = jnp.expand_dims(jnp.argmax(
                            jnp.sum(self.get_q(params,nxtobses,key)*self._categorial_bar,axis=2)
                            ,axis=1),axis=(1,2))
        else:
            next_actions = jnp.expand_dims(jnp.argmax(
                            jnp.sum(next_q*self._categorial_bar,axis=2)
                            ,axis=1),axis=(1,2))
        next_distribution = jnp.squeeze(jnp.take_along_axis(next_q, next_actions, axis=1))
        
        if self.munchausen:
            next_q_mean = jnp.sum(next_q*self.categorial_bar,axis=2)
            logsum = jax.nn.logsumexp((next_q_mean - jnp.max(next_q_mean,axis=1,keepdims=True))/self.munchausen_entropy_tau, axis=1, keepdims=True)
            tau_log_pi_next = next_q_mean - jnp.max(next_q_mean, axis=1, keepdims=True) - self.munchausen_entropy_tau*logsum
            pi_target = jax.nn.softmax(next_q_mean/self.munchausen_entropy_tau, axis=1)
            next_categorial = (self.categorial_bar - jnp.sum(pi_target * tau_log_pi_next, axis=1, keepdims=True)) * not_dones
            
            q_k_targets = jnp.sum(self.get_q(target_params,obses,key)*self.categorial_bar,axis=2)
            v_k_target = jnp.max(q_k_targets, axis=1, keepdims=True)
            logsum = jax.nn.logsumexp((q_k_targets - v_k_target)/self.munchausen_entropy_tau, axis=1, keepdims=True)
            log_pi = q_k_targets - v_k_target - self.munchausen_entropy_tau*logsum
            munchausen_addon = jnp.take_along_axis(log_pi,jnp.squeeze(actions,axis=2),axis=1)
            
            rewards += self.munchausen_alpha*jnp.clip(munchausen_addon, a_min=-1, a_max=0)
        else:
            next_categorial = not_dones * self.categorial_bar
        target_categorial = (next_categorial * self._gamma) + rewards
        Tz = jnp.clip( target_categorial, self.categorial_min,self.categorial_max)
        C51_b = ((Tz - self.categorial_min)/self.delta_bar).astype(jnp.float32)
        C51_L = jnp.floor(C51_b).astype(jnp.int32)
        C51_H = jnp.ceil(C51_b).astype(jnp.int32)
        C51_L = jnp.where((C51_H > 0) * (C51_L == C51_H), C51_L - 1, C51_L) #C51_L.at[].add(-1)
        C51_H = jnp.where((C51_L < (self.categorial_bar_n - 1)) * (C51_L == C51_H), C51_H + 1, C51_H) #C51_H.at[].add(1)
        target_distribution = jnp.zeros((self.batch_size*self.categorial_bar_n))
        target_distribution = target_distribution.at[jnp.reshape(C51_L + self.offset,(-1))].add(jnp.reshape(next_distribution*(C51_H.astype(jnp.float32) - C51_b),(-1)))
        target_distribution = target_distribution.at[jnp.reshape(C51_H + self.offset,(-1))].add(jnp.reshape(next_distribution*(C51_b - C51_L.astype(jnp.float32)),(-1)))
        target_distribution = jnp.reshape(target_distribution,(self.batch_size,self.categorial_bar_n))
        return jax.lax.stop_gradient(target_distribution)

    
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="C51",
              reset_num_timesteps=True, replay_wrapper=None):
        super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)