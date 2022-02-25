import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax

from haiku_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from haiku_baselines.TD4_QR.network import Actor, Critic
from haiku_baselines.common.Module import PreProcess

from haiku_baselines.common.utils import soft_update, convert_jax, truncated_mixture
from haiku_baselines.common.losses import QuantileHuberLosses

class TD4_QR(Deteministic_Policy_Gradient_Family):
    def __init__(self, env, gamma=0.995, learning_rate=3e-4, buffer_size=100000,target_action_noise_mul = 2.0, 
                 n_support = 200, delta = 1.0, action_noise = 0.1, train_freq=1, gradient_steps=1, batch_size=32, policy_delay = 3,
                 n_step = 1, learning_starts=1000, target_network_update_tau=5e-4, prioritized_replay=False, mixture_type = 'min',
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_eps=1e-6, 
                 log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None, optimizer = 'adamw'):
        
        super(TD4_QR, self).__init__(env, gamma, learning_rate, buffer_size, train_freq, gradient_steps, batch_size,
                 n_step, learning_starts, target_network_update_tau, prioritized_replay,
                 prioritized_replay_alpha, prioritized_replay_beta0, prioritized_replay_eps,
                 log_interval, tensorboard_log, _init_setup_model, policy_kwargs, 
                 full_tensorboard_log, seed, optimizer)
        
        self.action_noise = action_noise
        self.traget_action_noise = action_noise*target_action_noise_mul
        self.action_noise_clamp = 0.5 #self.target_action_noise*1.5
        self.policy_delay = policy_delay
        self.n_support = n_support
        self.mixture_type = mixture_type
        self.delta = delta
        
        if _init_setup_model:
            self.setup_model() 
            
    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
        if 'cnn_mode' in self.policy_kwargs.keys():
            cnn_mode = self.policy_kwargs['cnn_mode']
            del self.policy_kwargs['cnn_mode']
        self.preproc = hk.transform(lambda x: PreProcess(self.observation_space, cnn_mode=cnn_mode)(x))
        self.actor = hk.transform(lambda x: Actor(self.action_size,
                           **self.policy_kwargs)(x))
        self.critic = hk.transform(lambda x,a: Critic(support_n=self.n_support,**self.policy_kwargs)(x,a))
        pre_param = self.preproc.init(next(self.key_seq),
                            [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
        feature = self.preproc.apply(pre_param, None, [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
        actor_param = self.actor.init(next(self.key_seq), feature)
        critic_param = self.critic.init(next(self.key_seq), feature, np.zeros((1,self.action_size[0])))
        self.params = hk.data_structures.merge(pre_param, actor_param, critic_param)
        self.target_params = self.params
        
        self.opt_state = self.optimizer.init(self.params)
        
        self.quantile = (jnp.linspace(0.0,1.0,self.n_support+1,dtype=jnp.float32)[1:] + 
                         jnp.linspace(0.0,1.0,self.n_support+1,dtype=jnp.float32)[:-1]) / 2.0  # [support]
        self.quantile = jax.device_put(jnp.expand_dims(self.quantile,axis=(0,1))).astype(jnp.float32)  # [1 x 1 x support]
        
        print("----------------------model----------------------")
        print(jax.tree_map(lambda x: x.shape, pre_param))
        print(jax.tree_map(lambda x: x.shape, actor_param))
        print(jax.tree_map(lambda x: x.shape, critic_param))
        print("-------------------------------------------------")

        self._get_actions = jax.jit(self._get_actions)
        self._train_step = jax.jit(self._train_step)
        
    def _get_actions(self, params, obses, key = None) -> jnp.ndarray:
        return self.actor.apply(params, key, self.preproc.apply(params, key, convert_jax(obses))) #
    
    def discription(self):
        return "score : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque), np.mean(self.lossque)
                                    )
    
    def actions(self,obs,steps):
        actions = np.clip(np.asarray(self._get_actions(self.params,obs, None)) + 
                          np.random.normal(0,self.action_noise,size=(self.worker_size,self.action_size[0])),-1,1)
        return actions
    
    def train_step(self, steps, gradient_steps):
        # Sample a batch from the replay buffer
        for _ in range(gradient_steps):
            if self.prioritized_replay:
                data = self.replay_buffer.sample(self.batch_size,self.prioritized_replay_beta0)
            else:
                data = self.replay_buffer.sample(self.batch_size)
            
            self.params, self.target_params, self.opt_state, loss, t_mean, new_priorities = \
                self._train_step(self.params, self.target_params, self.opt_state, next(self.key_seq), steps,
                                 **data)
            
            if self.prioritized_replay:
                self.replay_buffer.update_priorities(data['indexes'], new_priorities)
            
        if self.summary and steps % self.log_interval == 0:
            self.summary.add_scalar("loss/qloss", loss, steps)
            self.summary.add_scalar("loss/targets", t_mean, steps)
            
        return loss

    def _train_step(self, params, target_params, opt_state, key, step,
                    obses, actions, rewards, nxtobses, dones, weights=1, indexes=None):
        obses = convert_jax(obses); nxtobses = convert_jax(nxtobses); not_dones = 1.0 - dones
        targets = self._target(target_params, rewards, nxtobses, not_dones, key)
        (total_loss, (critic_loss, actor_loss, abs_error)), grad = jax.value_and_grad(self._loss,has_aux = True)(params, obses, actions, targets, weights, key, step)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        target_params = soft_update(params, target_params, self.target_network_update_tau)
        new_priorities = None
        if self.prioritized_replay:
            new_priorities = abs_error + self.prioritized_replay_eps
        return params, target_params, opt_state, critic_loss, -actor_loss, new_priorities
    
    def _loss(self, params, obses, actions, targets, weights, key, step):
        feature = self.preproc.apply(params, key, obses)
        q1, q2 = self.critic.apply(params, key, feature, actions)
        q1_loss_tile = jnp.expand_dims(q1,axis=1)                                               # batch x 1 x support
        q2_loss_tile = jnp.expand_dims(q2,axis=1)                                               # batch x 1 x support
        logit_valid_tile = jnp.expand_dims(targets,axis=2)                                      # batch x support x 1
        huber1 = QuantileHuberLosses(q1_loss_tile, logit_valid_tile, self.quantile, self.delta)
        huber2 = QuantileHuberLosses(q2_loss_tile, logit_valid_tile, self.quantile, self.delta)
        critic_loss = jnp.mean(weights*huber1) + jnp.mean(weights*huber2)
        policy = self.actor.apply(params, key, feature)
        distributed_policy = jnp.clip(jnp.expand_dims(policy,axis=0)
                                        + self.action_noise*jax.random.normal(key,(5,self.batch_size,self.action_size[0]))
                                        ,-1.0,1.0)
        vals, _ = self.critic.apply(jax.lax.stop_gradient(params), key, feature, distributed_policy[0])
        actor_loss = -jnp.mean(vals)
        for dp in distributed_policy[1:]:
            vals, _ = self.critic.apply(jax.lax.stop_gradient(params), key, feature, dp)
            actor_loss += -jnp.mean(vals)
        total_loss = critic_loss + actor_loss
        return total_loss, (critic_loss, actor_loss, huber1)
    
    def _target(self, target_params, rewards, nxtobses, not_dones, key):
        next_feature = self.preproc.apply(target_params, key, nxtobses)
        next_action = jnp.clip(
                      self.actor.apply(target_params, key, next_feature) \
                      + jnp.clip(self.traget_action_noise*jax.random.normal(key,(self.batch_size,self.action_size[0])),-self.action_noise_clamp,self.action_noise_clamp)
                      ,-1.0,1.0)
        q1, q2 = self.critic.apply(target_params, key, next_feature, next_action)
        if self.mixture_type == 'min':
            next_q = jnp.minimum(q1,q2)
        elif self.mixture_type == 'truncated':
            next_q = truncated_mixture((q1, q2),self.n_support*2 - 2)
        
        return (not_dones * next_q * self._gamma) + rewards
    
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="TD4_QR",
              reset_num_timesteps=True, replay_wrapper=None):
        super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)