import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax

from haiku_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from haiku_baselines.A2C.network import Actor, Critic
from haiku_baselines.common.Module import PreProcess
from haiku_baselines.common.utils import convert_jax, discount_with_terminal, print_param

class A2C(Actor_Critic_Policy_Gradient_Family):
    def __init__(self, env, gamma=0.995, learning_rate=3e-4, batch_size=32, val_coef=0.2, ent_coef = 0.5,
                 log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None, optimizer = 'rmsprop'):
        
        super(A2C, self).__init__(env, gamma, learning_rate, batch_size, val_coef, ent_coef,
                 log_interval, tensorboard_log, _init_setup_model, policy_kwargs, 
                 full_tensorboard_log, seed, optimizer)
        
        self.get_memory_setup()

        if _init_setup_model:
            self.setup_model() 
            
    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
        if 'cnn_mode' in self.policy_kwargs.keys():
            cnn_mode = self.policy_kwargs['cnn_mode']
            del self.policy_kwargs['cnn_mode']
        self.preproc = hk.transform(lambda x: PreProcess(self.observation_space, cnn_mode=cnn_mode)(x))
        self.actor = hk.transform(lambda x: Actor(self.action_size,self.action_type,**self.policy_kwargs)(x))
        self.critic = hk.transform(lambda x: Critic(**self.policy_kwargs)(x))
        pre_param = self.preproc.init(next(self.key_seq),
                            [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
        feature = self.preproc.apply(pre_param, None, [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
        actor_param = self.actor.init(next(self.key_seq), feature)
        critic_param = self.critic.init(next(self.key_seq), feature)
        self.params = hk.data_structures.merge(pre_param, actor_param, critic_param)
        
        self.opt_state = self.optimizer.init(self.params)
        
        print("----------------------model----------------------")
        print_param('preprocess',pre_param)
        print_param('actor',actor_param)
        print_param('critic',critic_param)
        print("-------------------------------------------------")
        
        self._get_actions = jax.jit(self._get_actions)
        self._train_step = jax.jit(self._train_step)
        
    def _get_actions_discrete(self, params, obses, key = None) -> jnp.ndarray:
        prob = jax.nn.softmax(self.actor.apply(params, key, self.preproc.apply(params, key, convert_jax(obses))),axis=1,)
        return prob
    
    def _get_actions_continuous(self, params, obses, key = None) -> jnp.ndarray:
        mu,std = self.actor.apply(params, key, self.preproc.apply(params, key, convert_jax(obses)))
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
            return prob, - (0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-6)),axis=-1,keepdims=True) + 
                                   jnp.sum(log_std,axis=-1,keepdims=True) + 
                                   0.5 * jnp.log(2 * np.pi)* jnp.asarray(action.shape[-1],dtype=jnp.float32))
        else:
            return - (0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-6)),axis=-1,keepdims=True) + 
                             jnp.sum(log_std,axis=-1,keepdims=True) + 
                             0.5 * jnp.log(2 * np.pi)* jnp.asarray(action.shape[-1],dtype=jnp.float32))
    
    
    def discription(self):
        return "score : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque), np.mean(self.lossque)
                                    )
    
    def action_discrete(self,obs):
        prob = np.asarray(self._get_actions(self.params, obs))
        return np.expand_dims(np.stack([np.random.choice(self.action_size[0],p=p) for p in prob],axis=0),axis=1)
    
    def action_continuous(self,obs):
        mu, std = self._get_actions(self.params, obs)
        return np.random.normal(np.array(mu), np.array(std))
    
    def train_step(self, steps):
        # Sample a batch from the replay buffer
        data = self.buffer.get_buffer()
        
        self.params, self.opt_state, critic_loss, actor_loss = \
            self._train_step(self.params, self.opt_state, None, self.ent_coef,
                                **data)
            
        if self.summary:
            self.summary.add_scalar("loss/critic_loss", critic_loss, steps)
            self.summary.add_scalar("loss/actor_loss", actor_loss, steps)
            
        return critic_loss

    def _train_step(self, params, opt_state, key, ent_coef,
                    obses, actions, rewards, nxtobses, dones, terminals):
        obses = [convert_jax(o) for o in obses]; nxtobses = [convert_jax(n) for n in nxtobses]
        value = [self.critic.apply(params, key, self.preproc.apply(params, key, o)) for o in obses]
        next_value = [self.critic.apply(params, key, self.preproc.apply(params, key, n)) for n in nxtobses]
        targets = [discount_with_terminal(r,d,t,nv,self.gamma) for r,d,t,nv in zip(rewards,dones,terminals,next_value)]
        obses = [jnp.vstack(list(zo)) for zo in zip(*obses)]; actions = jnp.vstack(actions);
        value = jnp.vstack(value); targets = jnp.vstack(targets); adv = targets - value
        (total_loss, (critic_loss, actor_loss)), grad = jax.value_and_grad(self._loss,has_aux = True)(params, 
                                                        obses, actions, targets, adv, ent_coef, key)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, critic_loss, actor_loss
    
    def _loss_discrete(self, params, obses, actions, targets, adv, ent_coef, key):
        feature = self.preproc.apply(params, key, obses)
        vals = self.critic.apply(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))
        
        prob, log_prob = self.get_logprob(self.actor.apply(params, key, feature), actions, key, out_prob=True)
        actor_loss = -jnp.mean(log_prob*adv)
        entropy = prob * jnp.log(prob)
        entropy_loss = jnp.mean(entropy)
        total_loss = self.val_coef * critic_loss + actor_loss - ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss)
    
    def _loss_continuous(self, params, obses, actions, targets, adv, ent_coef, key):
        feature = self.preproc.apply(params, key, obses)
        vals = self.critic.apply(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))
        
        prob, log_prob = self.get_logprob(self.actor.apply(params, key, feature), actions, key, out_prob=True)
        actor_loss = -jnp.mean(log_prob*adv)
        mu, log_std = prob
        entropy_loss = jnp.mean(0.5 + 0.5 * jnp.log(2 * np.pi) + log_std) #jnp.mean(jnp.square(mu) + jnp.square(log_std))
        total_loss = self.val_coef * critic_loss + actor_loss - ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss)
    
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="A2C",
              reset_num_timesteps=True, replay_wrapper=None):
        
        super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)