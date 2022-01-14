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
    def __init__(self, env, gamma=0.99, lamda = 0.95, gae_normalize = True, learning_rate=3e-4, batch_size=32, ent_coef = 0.5,
                 log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None, optimizer = 'adamw'):
        
        super(A2C, self).__init__(env, gamma, lamda, gae_normalize, learning_rate, batch_size, ent_coef,
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
        self.critic = hk.transform(lambda x: Critic(**self.policy_kwargs)(x))
        pre_param = self.preproc.init(next(self.key_seq),
                            [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
        feature = self.preproc.apply(pre_param, None, [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
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
        #self._train_step = jax.jit(self._train_step)
        
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
    
    def train_step(self, steps):
        # Sample a batch from the replay buffer
        data = self.buffer.get_buffer()
        
        self.params, self.opt_state, critic_loss, actor_loss = \
            self._train_step(self.params, self.opt_state, None, self.ent_coef,
                                **data)
            
        if self.summary and steps % self.log_interval == 0:
            self.summary.add_scalar("loss/critic_loss", critic_loss, steps)
            self.summary.add_scalar("loss/actor_loss", actor_loss, steps)
            
        return critic_loss

    def _train_step(self, params, opt_state, key, ent_coef,
                    obses, actions, rewards, nxtobses, dones):
        obses = [convert_jax(o) for o in obses]; nxtobses = [convert_jax(n) for n in nxtobses]
        value = [self.critic.apply(params, key, self.preproc.apply(params, None, o)) for o in obses]
        next_value = [self.critic.apply(params, key, self.preproc.apply(params, None, n)) for n in nxtobses]
        adv, targets = zip(*[get_gaes(r, d, v, nv, self.gamma, self.lamda, self.gae_normalize) for r, d, v, nv in zip(rewards, dones, value, next_value)])
        obses_hstack = [jnp.hstack(zo) for zo in list(zip(*obses))]
        action_hstack = jnp.hstack(actions)
        adv_hstack = jnp.hstack(adv)
        target_hstack = jnp.hstack(targets)
        '''
        for oh in obses_hstack:
            print('ob :', oh.shape, ', ', oh)
            
        print('act: ', action_hstack.shape, ', ', action_hstack)
        print('adv: ', adv_hstack.shape, ', ', adv_hstack)
        print('target: ', target_hstack.shape, ', ', target_hstack)
        '''
        (total_loss, (critic_loss, actor_loss)), grad = jax.value_and_grad(self._loss,has_aux = True)(params, 
                                                        obses_hstack, action_hstack, adv_hstack, target_hstack, ent_coef, key)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, critic_loss, actor_loss
    
    def _loss_discrete(self, params, obses, actions, targets, adv, ent_coef, key):
        feature = self.preproc.apply(params, key, obses)
        vals = self.critic.apply(params, key, feature)
        error = jnp.squeeze(vals - targets)
        critic_loss = jnp.mean(jnp.square(error))
        prob = jnp.clip(jax.nn.softmax(self.actor.apply(params, key, feature)),13-5,1.0)
        action_prob = jnp.take_along_axis(prob, actions, axis=1)
        cross_entropy = action_prob*adv
        actor_loss = -jnp.mean(cross_entropy)
        entropy = prob * jnp.log(prob)
        entropy_loss = jnp.mean(entropy)
        total_loss = critic_loss + actor_loss - ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss)
    
    def _loss_continuous(self, params, obses, actions, targets, adv, ent_coef, key):
        pass
    
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DDPG",
              reset_num_timesteps=True, replay_wrapper=None):
        
        super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)