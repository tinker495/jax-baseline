import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax

from haiku_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from haiku_baselines.PPO.network import Actor, Critic
from haiku_baselines.common.schedules import LinearSchedule
from haiku_baselines.common.Module import PreProcess
from haiku_baselines.common.utils import convert_jax, get_gaes

class PPO(Actor_Critic_Policy_Gradient_Family):
    def __init__(self, env, gamma=0.99, lamda = 0.9, gae_normalize = False, learning_rate=3e-4, batch_size=512, minibatch_size=16, val_coef=0.2, ent_coef = 0.5, 
                 clip_value = 100.0, ppo_eps = 0.2, log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None, optimizer = 'rmsprop'):
        
        super(PPO, self).__init__(env, gamma, lamda, gae_normalize, learning_rate, batch_size, val_coef, ent_coef,
                 log_interval, tensorboard_log, _init_setup_model, policy_kwargs, 
                 full_tensorboard_log, seed, optimizer)
        
        self.ppo_eps = ppo_eps
        self.minibatch_size = minibatch_size
        self.clip_value = clip_value
        
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
        self._train_step = jax.jit(self._train_step)
        
    def _get_actions_discrete(self, params, obses, key = None) -> jnp.ndarray:
        prob = jax.nn.softmax(self.actor.apply(params, None, self.preproc.apply(params, None, convert_jax(obses))),axis=1)
        return prob
    
    def _get_actions_continuous(self, params, obses, key = None) -> jnp.ndarray:
        mu,std = self.actor.apply(params, None, self.preproc.apply(params, None, convert_jax(obses)))
        return mu, std
    
    def discription(self):
        return "score : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque), np.mean(self.lossque)
                                    )
    
    def action_discrete(self,obs,steps):
        prob = self._get_actions(self.params, obs)
        return np.expand_dims(np.stack([np.random.choice(self.action_size[0],p=p) for p in prob],axis=0),axis=1)
    
    def action_continuous(self,obs,steps):
        mu, std = self._get_actions(self.params, obs)
        return mu
    
    def train_step(self, steps):
        # Sample a batch from the replay buffer
        data = self.buffer.get_buffer()
        
        self.params, self.opt_state, critic_loss, actor_loss = \
            self._train_step(self.params, self.opt_state, next(self.key_seq), self.ent_coef,
                                **data)
            
        #print(np.mean(adv_hstack))
        #print(np.mean(target_hstack))
        if self.summary:
            self.summary.add_scalar("loss/critic_loss", critic_loss, steps)
            self.summary.add_scalar("loss/actor_loss", actor_loss, steps)
            
        return critic_loss

    def _train_step(self, params, opt_state, key, ent_coef,
                    obses, actions, rewards, nxtobses, dones, terminals):
        obses = [convert_jax(o) for o in obses]; nxtobses = [convert_jax(n) for n in nxtobses]
        value = [self.critic.apply(params, key, self.preproc.apply(params, key, o)) for o in obses]
        act_prob = [jnp.take_along_axis(jnp.clip(jax.nn.softmax(self.actor.apply(params, key, self.preproc.apply(params, key, o))),1e-5,1.0), a, axis=1) for o,a in zip(obses,actions)]
        next_value = [self.critic.apply(params, key, self.preproc.apply(params, key, n)) for n in nxtobses]
        adv,targets = list(zip(*[get_gaes(r, d, t, v, nv, self.gamma, self.lamda) for r,d,t,v,nv in zip(rewards,dones,terminals,value,next_value)]))
        obses = [jnp.vstack(zo) for zo in list(zip(*obses))]; actions = jnp.vstack(actions); value = jnp.vstack(value); act_prob = jnp.vstack(act_prob)
        targets = jnp.vstack(targets); adv = jnp.vstack(adv); 
        adv = (adv - jnp.mean(adv,keepdims=True)) / (jnp.std(adv,keepdims=True) + 1e-8)
        def f(update_state , info):
            params, opt_state = update_state
            obses, actions, targets, value, act_prob, adv = info
            (total_loss, (c_loss, a_loss)), grad = jax.value_and_grad(self._loss,has_aux = True)(params, 
                                                        obses, actions, targets,
                                                        value, act_prob, adv, ent_coef, key)
            updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), (c_loss, a_loss)
        
        batched_obses = list(zip(*[jnp.split(o, self.minibatch_size) for o in obses]))
        for bb in batched_obses:
            for b in bb:
                print(b.shape)
        batched_actions = jnp.split(actions, self.minibatch_size)
        batched_targets = jnp.split(targets, self.minibatch_size)
        batched_value = jnp.split(value, self.minibatch_size)
        batched_act_prob = jnp.split(act_prob, self.minibatch_size)
        batched_adv = jnp.split(adv, self.minibatch_size)
        (params, opt_state), (critic_loss, actor_loss) = \
                        jax.lax.scan(f,(params, opt_state),(batched_obses, batched_actions, batched_targets, batched_value, batched_act_prob, batched_adv))

        return params, opt_state, jnp.mean(critic_loss), jnp.mean(actor_loss)
    
    def _loss_discrete(self, params, obses, actions, targets, old_value, old_prob, adv, ent_coef, key):
        feature = self.preproc.apply(params, key, obses)
        vals = self.critic.apply(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))
        
        prob = jnp.clip(jax.nn.softmax(self.actor.apply(params, key, feature)),1e-5,1.0)
        action_prob = jnp.take_along_axis(prob, actions, axis=1)
        ratio = jnp.exp(jnp.log(action_prob) - jnp.log(old_prob))
        cross_entropy1 = adv*ratio; cross_entropy2 = adv*jnp.clip(ratio,1 - self.ppo_eps,1 + self.ppo_eps)
        actor_loss = -jnp.mean(jnp.minimum(cross_entropy1,cross_entropy2))
        entropy = prob * jnp.log(prob)
        entropy_loss = jnp.mean(entropy)
        total_loss = self.val_coef * critic_loss + actor_loss - ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss)
    
    def _loss_continuous(self, params, obses, actions, targets, adv, ent_coef, key):
        pass
    
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="PPO",
              reset_num_timesteps=True, replay_wrapper=None):
        
        super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)