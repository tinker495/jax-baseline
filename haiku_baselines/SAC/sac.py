import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax

from haiku_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from haiku_baselines.SAC.network import Actor, Critic, Value

from haiku_baselines.common.Module import PreProcess
from haiku_baselines.common.utils import soft_update, convert_jax, print_param

class SAC(Deteministic_Policy_Gradient_Family):
    def __init__(self, env, gamma=0.995, learning_rate=3e-4, buffer_size=100000, train_freq=1, gradient_steps=1, ent_coef = 'auto', 
                 batch_size=32, policy_delay = 3, n_step = 1, learning_starts=1000, target_network_update_tau=5e-4,
                 prioritized_replay=False, prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4,
                 prioritized_replay_eps=1e-6, log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None, optimizer = 'adamw'):
        
        super(SAC, self).__init__(env, gamma, learning_rate, buffer_size, train_freq, gradient_steps, batch_size,
                 n_step, learning_starts, target_network_update_tau, prioritized_replay,
                 prioritized_replay_alpha, prioritized_replay_beta0, prioritized_replay_eps,
                 log_interval, tensorboard_log, _init_setup_model, policy_kwargs, 
                 full_tensorboard_log, seed, optimizer)
        
        self.policy_delay = policy_delay
        self.ent_coef = ent_coef
        self.target_entropy = -np.prod(self.action_size).astype(np.float32) #
        self.ent_coef_learning_rate = 1e-6
        
        if _init_setup_model:
            self.setup_model() 
            
    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
        if 'cnn_mode' in self.policy_kwargs.keys():
            cnn_mode = self.policy_kwargs['cnn_mode']
            del self.policy_kwargs['cnn_mode']
        self.preproc = hk.transform(lambda x: PreProcess(self.observation_space, cnn_mode=cnn_mode)(x))
        self.actor = hk.transform(lambda x: Actor(self.action_size,**self.policy_kwargs)(x))
        self.critic = hk.transform(lambda x,a: Critic(**self.policy_kwargs)(x,a))
        self.value = hk.transform(lambda x: Value(**self.policy_kwargs)(x))
        pre_param = self.preproc.init(next(self.key_seq),
                            [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
        feature = self.preproc.apply(pre_param, None, [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
        actor_param = self.actor.init(next(self.key_seq), feature)
        critic_param = self.critic.init(next(self.key_seq), feature, np.zeros((1,self.action_size[0])))
        value_param = self.value.init(next(self.key_seq), feature)
        self.params = hk.data_structures.merge(pre_param, actor_param, critic_param, value_param)
        self.target_params = self.params
        
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
            init_value = np.log(1e-1)
            if '_' in self.ent_coef:
                init_value = np.log(float(self.ent_coef.split('_')[1]))
                assert init_value > 0., "The initial value of ent_coef must be greater than 0"
            self.log_ent_coef = jax.device_put(init_value)
            self.ent_coef = jnp.exp(self.log_ent_coef)
        else:
            self.ent_coef = float(self.ent_coef)
        
        self.opt_state = self.optimizer.init(self.params)
        
        print("----------------------model----------------------")
        print_param('preprocess',pre_param)
        print_param('actor',actor_param)
        print_param('critic',critic_param)
        print_param('value',value_param)
        print("-------------------------------------------------")

        self._get_actions = jax.jit(self._get_actions)
        self._train_step = jax.jit(self._train_step)
        self._train_ent_coef = jax.jit(self._train_ent_coef)
        
    def _get_update_data(self,params,feature,key = None) -> jnp.ndarray:
        mu, log_std = self.actor.apply(params, None, feature)
        std = jnp.exp(log_std)
        x_t = mu + std * jax.random.normal(key,std.shape)
        pi = jax.nn.tanh(x_t)
        log_prob = jnp.sum(-0.5 * (
            jnp.square((x_t - mu) / (std + 1e-6))
            + 2 * log_std
            + jnp.log(2 * np.pi)
            ) - jnp.log(1 - jnp.square(pi) + 1e-6),axis=1,keepdims=True)
        return pi, log_prob, mu, log_std, std
        
    def _get_actions(self, params, obses, key = None) -> jnp.ndarray:
        mu, log_std = self.actor.apply(params, None, self.preproc.apply(params, None, convert_jax(obses)))
        std = jnp.exp(log_std)
        pi = jax.nn.tanh(mu + std * jax.random.normal(key,std.shape))
        return pi
    
    def discription(self):
        return "score : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque), np.mean(self.lossque)
                                    )
    
    def actions(self,obs,steps):
        if self.learning_starts < steps:
            actions = np.asarray(self._get_actions(self.params,obs, next(self.key_seq)))
        else:
            actions = np.random.uniform(-1.0,1.0,size=(self.worker_size,self.action_size[0]))
        return actions
    
    def train_step(self, steps, gradient_steps):
        # Sample a batch from the replay buffer
        for _ in range(gradient_steps):
            if self.prioritized_replay:
                data = self.replay_buffer.sample(self.batch_size,self.prioritized_replay_beta0)
            else:
                data = self.replay_buffer.sample(self.batch_size)
            
            self.params, self.target_params, self.opt_state, loss, t_mean, log_prob,new_priorities = \
                self._train_step(self.params, self.target_params, self.opt_state, next(self.key_seq), steps, self.ent_coef,
                                 **data)
            
            if not isinstance(self.ent_coef, float):
                self.log_ent_coef, self.ent_coef = self._train_ent_coef(self.log_ent_coef, log_prob)
                
            if self.prioritized_replay:
                self.replay_buffer.update_priorities(data['indexes'], new_priorities)
            
        if self.summary and steps % self.log_interval == 0:
            self.summary.add_scalar("loss/qloss", loss, steps)
            self.summary.add_scalar("loss/targets", t_mean, steps)
            self.summary.add_scalar("loss/ent_coef", self.ent_coef, steps)
            
        return loss

    def _train_step(self, params, target_params, opt_state, key, step, ent_coef,
                    obses, actions, rewards, nxtobses, dones, weights=1, indexes=None):
        obses = convert_jax(obses); nxtobses = convert_jax(nxtobses); not_dones = 1.0 - dones
        targets = self._target(target_params, rewards, nxtobses, not_dones, key)
        (total_loss, (critic_loss, actor_loss, abs_error, log_prob)), grad = jax.value_and_grad(self._loss,has_aux = True)(params, obses, actions, targets, weights, key, step, ent_coef)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        target_params = soft_update(params, target_params, self.target_network_update_tau)
        new_priorities = None
        if self.prioritized_replay:
            new_priorities = abs_error + self.prioritized_replay_eps
        return params, target_params, opt_state, critic_loss, -actor_loss, log_prob, new_priorities
    
    def _train_ent_coef(self,log_coef,log_prob):
        l = lambda log_ent_coef, log_prob: -jnp.mean(log_ent_coef * (log_prob + self.target_entropy))
        grad = jax.grad(l)(log_coef,log_prob)
        log_coef = log_coef - self.ent_coef_learning_rate * grad
        return log_coef, jnp.exp(log_coef)
    
    def _loss(self, params, obses, actions, targets, weights, key, step, ent_coef):
        feature = self.preproc.apply(params, key, obses)
        policy, log_prob, mu, log_std, std = self._get_update_data(params, feature, key)
        q1, q2 = self.critic.apply(params, key, feature, actions)
        q1_pi, q2_pi = self.critic.apply(jax.lax.stop_gradient(params), key, feature, policy)
        value = self.value.apply(params, key, feature)
        error1 = jnp.squeeze(q1 - targets)
        error2 = jnp.squeeze(q2 - targets)
        error_v = jnp.squeeze(value - jax.lax.stop_gradient(jnp.minimum(q1_pi,q2_pi) - ent_coef * log_prob))
        critic_loss = jnp.mean(weights*jnp.square(error1)) + jnp.mean(weights*jnp.square(error2)) + jnp.mean(jnp.square(error_v))
        actor_loss = jnp.mean(ent_coef * log_prob - jnp.minimum(q1_pi,q2_pi))
        total_loss = jax.lax.select(step % self.policy_delay == 0, critic_loss + actor_loss, critic_loss)
        return total_loss, (critic_loss, actor_loss, jnp.abs(error_v), log_prob)
    
    def _target(self, target_params, rewards, nxtobses, not_dones, key):
        next_feature = self.preproc.apply(target_params, key, nxtobses)
        v = self.value.apply(target_params, key, next_feature)
        return (not_dones * v * self._gamma) + rewards
    
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="SAC",
              reset_num_timesteps=True, replay_wrapper=None):
        super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)