import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax

from haiku_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from haiku_baselines.A2C.network import Actor, Critic
from haiku_baselines.common.schedules import LinearSchedule
from haiku_baselines.common.Module import PreProcess
from haiku_baselines.common.utils import convert_jax, get_gaes

class A2C(Actor_Critic_Policy_Gradient_Family):
    def __init__(self, env, gamma=0.99, learning_rate=5e-5, gradient_steps=1, batch_size=32,
                 log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None, optimizer = 'adamw'):
        
        super(A2C, self).__init__(env, gamma, learning_rate, gradient_steps, batch_size,
                 log_interval, tensorboard_log, _init_setup_model, policy_kwargs, 
                 full_tensorboard_log, seed, optimizer)
        
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
        pre_param = self.preproc.init(next(self.key_seq),
                            [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
        feature = self.preproc.apply(pre_param, None, [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
        actor_param = self.actor.init(next(self.key_seq), feature)
        critic_param = self.critic.init(next(self.key_seq), feature, np.zeros((1,self.action_size[0])))
        self.params = hk.data_structures.merge(pre_param, actor_param, critic_param)
        
        self.opt_state = self.optimizer.init(self.params)
        
        print("----------------------model----------------------")
        print(jax.tree_map(lambda x: x.shape, pre_param))
        print(jax.tree_map(lambda x: x.shape, actor_param))
        print(jax.tree_map(lambda x: x.shape, critic_param))
        print("-------------------------------------------------")
        
        self._get_actions = jax.jit(self._get_actions)
        self._train_step = jax.jit(self._train_step)
        
    def _get_actions_discrete(self, params, obses, key = None) -> jnp.ndarray:
        prob = jax.nn.softmax(self.actor.apply(params, None, self.preproc.apply(params, None, convert_jax(obses))),axis=1)
        return prob
    
    def _get_actions_continuous(self, params, obses, key = None) -> jnp.ndarray:
        mu,std = self.actor.apply(params, None, self.preproc.apply(params, None, convert_jax(obses)))
        return mu, std
    
    def discription(self):
        return "score : {:.3f}, epsilon : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque), self.epsilon, np.mean(self.lossque)
                                    )
    
    def action_discrete(self,obs,steps):
        prob = self._get_actions(self.params, obs)
        return np.stack([np.random.choice(self.action_size[0],p=p) for p in prob],axis=0)
    
    def action_continuous(self,obs,steps):
        mu, std = self._get_actions(self.params, obs)
        return mu
    
    def test_action(self, state):
        return np.clip(np.asarray(self._get_actions(self.params,state, None)) + self.noise()*self.exploration_final_eps,-1,1)
    
    def train_step(self, steps, gradient_steps):
        # Sample a batch from the replay buffer
        data = self.buffer.get_buffer()
        for _ in range(gradient_steps):
            
            self.params, self.target_params, self.opt_state, loss, t_mean = \
                self._train_step(self.params, self.target_params, self.opt_state, None,
                                 **data)
            
        if self.summary and steps % self.log_interval == 0:
            self.summary.add_scalar("loss/qloss", loss, steps)
            self.summary.add_scalar("loss/targets", t_mean, steps)
            
        return loss

    def _train_step_discrete(self, params, opt_state, key, 
                    obses, actions, rewards, nxtobses, dones, weights=1, indexes=None):
        obses = [convert_jax(o) for o in obses]; nxtobses = [convert_jax(n) for n in nxtobses]; not_dones = [1.0 - d for d in dones]
        value = [self.critic.apply(params, key, self.preproc.apply(params, None, o)) for o in obses]
        next_value = [self.critic.apply(params, key, self.preproc.apply(params, None, n)) for n in nxtobses]
        adv, targets = zip(*[get_gaes(r, d, v, nv, self.gamma, self.lamda, self.gae_normalize) for r, d, v, nv in zip(rewards, dones, value, next_value)])
        return 
    
    def _train_step_continuous(self, params, target_params, opt_state, key, 
                    obses, actions, rewards, nxtobses, dones, weights=1, indexes=None):
        obses = convert_jax(obses); nxtobses = convert_jax(nxtobses); not_dones = 1.0 - dones
        return params, target_params, opt_state, critic_loss, actor_loss
    
    def _critic_loss(self, params, obses, actions, targets, weights, key):
        feature = self.preproc.apply(params, key, obses)
        vals = self.critic.apply(params, key, feature, actions)
        error = jnp.squeeze(vals - targets)
        return jnp.mean(weights*jnp.square(error)), jnp.abs(error)
    
    def _actor_loss(self, params, obses, key):
        feature = self.preproc.apply(params, key, obses)
        policy = self.actor.apply(params, key, feature)
        vals = self.critic.apply(jax.lax.stop_gradient(params), key, feature, policy)
        return jnp.mean(-vals)
    
    def _target(self, target_params, rewards, nxtobses, not_dones, key):
        next_feature = self.preproc.apply(target_params, key, nxtobses)
        next_action = self.actor.apply(target_params, key, next_feature)
        next_q = self.critic.apply(target_params, key, next_feature, next_action)
        return (not_dones * next_q * self._gamma) + rewards
    
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DDPG",
              reset_num_timesteps=True, replay_wrapper=None):
        
        self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                                initial_p=self.exploration_initial_eps,
                                                final_p=self.exploration_final_eps)
        self.epsilon = 1.0
        super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)