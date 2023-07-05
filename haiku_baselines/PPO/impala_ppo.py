import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from itertools import repeat

from haiku_baselines.IMPALA.base_class import IMPALA_Family
from haiku_baselines.PPO.network import Actor, Critic
from haiku_baselines.common.Module import PreProcess
from haiku_baselines.common.utils import get_vtrace, print_param

class IMPALA_PPO(IMPALA_Family):
    def __init__(self, workers, manager=None, buffer_size=0, gamma=0.995, lamda=0.95, learning_rate=0.0003, update_freq = 100, batch_size=1024, sample_size=1, epoch_num=3, val_coef=0.2, ent_coef=0.01, rho_max=1.0,
                 ppo_eps = 0.2, log_interval=1, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, optimizer='adamw'):
        super().__init__(workers, manager, buffer_size, gamma, lamda, learning_rate, update_freq, batch_size, sample_size, val_coef, ent_coef, rho_max, 
                         log_interval, tensorboard_log, _init_setup_model, policy_kwargs, full_tensorboard_log, seed, optimizer)
        
        self.ppo_eps = ppo_eps
        self.epoch_num = epoch_num
        self.get_memory_setup()

        if _init_setup_model:
            self.setup_model() 
        
    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
        if 'cnn_mode' in self.policy_kwargs.keys():
            cnn_mode = self.policy_kwargs['cnn_mode']
            del self.policy_kwargs['cnn_mode']
            
        def network_builder(observation_space, cnn_mode, action_size, action_type, **kwargs):
            def builder():
                preproc = hk.transform(lambda x: PreProcess(observation_space, cnn_mode=cnn_mode)(x))
                actor = hk.transform(lambda x: Actor(action_size,action_type,**kwargs)(x))
                critic = hk.transform(lambda x: Critic(**kwargs)(x))
                return preproc, actor, critic
            return builder
        self.network_builder = network_builder(self.observation_space, cnn_mode, self.action_size, self.action_type, **self.policy_kwargs)
        self.actor_builder = self.get_actor_builder()

        self.preproc, self.actor, self.critic = self.network_builder()
        pre_param = self.preproc.init(next(self.key_seq),
                            [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
        actor_param = self.actor.init(next(self.key_seq),
                            self.preproc.apply(pre_param, 
                            None, [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space]))
        critic_param = self.critic.init(next(self.key_seq),
                            self.preproc.apply(pre_param, 
                            None, [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space]))
        self.params = hk.data_structures.merge(pre_param, actor_param, critic_param)

        self.opt_state = self.optimizer.init(self.params)

        print("----------------------model----------------------")
        print_param('preprocess',pre_param)
        print_param('actor',actor_param)
        print_param('critic',critic_param)
        print("-------------------------------------------------")

        self._train_step = jax.jit(self._train_step)
        self.preprocess = jax.jit(self.preprocess)
        self._loss = jax.jit(self._loss_discrete) if self.action_type == 'discrete' else jax.jit(self._loss_continuous)

    def get_actor_builder(self):
        action_type = self.action_type
        def builder():
            key_seq = hk.PRNGSequence(42)
            
            if action_type == 'discrete':
                def actor(actor_model, preproc, params, obses, key):
                    prob = actor_model.apply(params, key, preproc.apply(params, key, convert_jax(obses)))
                    return prob
                
                def get_action_prob(actor, params, obses, key):
                    prob = actor(params, obses, key)
                    action = jax.random.categorical(key, prob)
                    prob = jnp.clip(jax.nn.softmax(prob), 1e-5, 1.0)
                    return action, jnp.log(jnp.take_along_axis(prob, jnp.expand_dims(action,axis=1), axis=1))
                
                def convert_action(action):
                    return int(action)
                
            elif action_type == 'continuous':
                def actor(actor_model, preproc, params, obses, key):
                    mean, log_std = actor_model.apply(params, key, preproc.apply(params, key, convert_jax(obses)))
                    return mean, log_std
                
                def get_action_prob(actor, params, obses, key):
                    mean, log_std = actor(params, obses, key)
                    std = jnp.exp(log_std)
                    action = jax.random.normal(key, mean.shape) * std + mean
                    return action, - (0.5 * jnp.sum(jnp.square((action - mean) / (std + 1e-7)),axis=-1,keepdims=True) + 
                                   jnp.sum(log_std,axis=-1,keepdims=True) + 
                                   0.5 * jnp.log(2 * np.pi)* jnp.asarray(action.shape[-1],dtype=jnp.float32))
                
                def convert_action(action):
                    return np.clip(action[0], -3.0, 3.0) / 3.0

            return actor, get_action_prob, convert_action, key_seq
        return builder

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
            return prob, - (0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-7)),axis=-1,keepdims=True) + 
                                   jnp.sum(log_std,axis=-1,keepdims=True) + 
                                   0.5 * jnp.log(2 * np.pi)* jnp.asarray(action.shape[-1],dtype=jnp.float32))
        else:
            return - (0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-7)),axis=-1,keepdims=True) + 
                             jnp.sum(log_std,axis=-1,keepdims=True) + 
                             0.5 * jnp.log(2 * np.pi)* jnp.asarray(action.shape[-1],dtype=jnp.float32))

    def train_step(self, steps):
        data = self.buffer.sample(self.sample_size)


        obses, actions, targets, old_prob, rho, adv = self.preprocess(self.params, next(self.key_seq), data[0], data[1], data[2], data[3], data[4], data[5], data[6])

        for i in range(self.epoch_num):
            self.params, self.opt_state, critic_loss, actor_loss, entropy_loss = \
            self._train_step(self.params, self.opt_state, next(self.key_seq), self.ent_coef, obses, actions, targets, old_prob, adv)
                    
        if steps % self.log_interval == 0:
            log_dict = {"loss/critic_loss": float(critic_loss), "loss/actor_loss": float(actor_loss), "loss/entropy_loss": float(entropy_loss),
                        "loss/mean_rho": float(jnp.mean(rho)), "loss/mean_target": float(jnp.mean(targets))}
            self.logger_server.log_trainer.remote(steps, log_dict)
        return critic_loss, float(jnp.mean(rho))
    
    def preprocess(self, params, key, obses, actions, mu_log_prob, rewards, nxtobses, dones, terminals):
        obses = [convert_jax(o) for o in obses]; nxtobses = [convert_jax(n) for n in nxtobses]
        feature = [self.preproc.apply(params, key, o) for o in obses]
        value = [self.critic.apply(params, key, f) for f in feature]
        next_value = [self.critic.apply(params, key, self.preproc.apply(params, key, n)) for n in nxtobses]
        pi_prob = [self.get_logprob(self.actor.apply(params, key, f), a, key) for f,a in zip(feature, actions)]
        rho_raw = [jnp.exp(pi - mu) for pi,mu in zip(pi_prob, mu_log_prob)]
        rho = [jnp.minimum(p, self.rho_max) for p in rho_raw]
        c_t = [self.lamda * jnp.minimum(p, self.cut_max) for p in rho_raw]
        vs = [get_vtrace(rw, p, c, d, t, v, nv, self.gamma) for rw, p, c, d, t, v, nv in zip(rewards, rho, c_t, dones, terminals, value, next_value)]
        vs_t_plus_1 = [jnp.concatenate([v[1:],jnp.expand_dims(nv[-1],axis=-1)]) for v,nv in zip(vs,next_value)]
        vs_t_plus_1 = [jnp.where(t==1, nv, vp) for t,nv,vp in zip(terminals,next_value,vs_t_plus_1)]
        adv = [p*(r + self.gamma * (1. - d) * nv - v) for p,r,d,nv,v in zip(rho, rewards, dones, vs_t_plus_1, value)]
        obses = [jnp.vstack(list(zo)) for zo in zip(*obses)]; 
        actions = jnp.vstack(actions); vs = jnp.vstack(vs); old_prob = jnp.vstack(pi_prob); rho = jnp.vstack(rho); adv = jnp.vstack(adv)
        #adv = (adv - jnp.mean(adv,keepdims=True)) / (jnp.std(adv,keepdims=True) + 1e-6)
        return obses, actions, vs, old_prob, rho, adv
    
    def _train_step(self, params, opt_state, key, ent_coef, obses, actions, targets, old_prob, adv):
        (total_loss, (critic_loss, actor_loss, entropy_loss)), grad = jax.value_and_grad(self._loss,has_aux = True)(params, obses, actions, targets, old_prob, adv, ent_coef, key)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, critic_loss, actor_loss, entropy_loss

    def _loss_discrete(self, params, obses, actions, vs, old_prob, adv, ent_coef, key):
        feature = self.preproc.apply(params, key, obses)
        vals = self.critic.apply(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(jax.lax.stop_gradient(vs) - vals))) #is sooooo large!!!
        #critic_loss =  jnp.mean(hubberloss(vs - vals, 1.0))
        
        logit = self.actor.apply(params, key, feature)
        prob, log_prob = self.get_logprob(logit, actions, key, out_prob=True)
        ratio = jnp.exp(log_prob - old_prob)
        cross_entropy1 = -adv*ratio; cross_entropy2 = -adv*jnp.clip(ratio,1.0 - self.ppo_eps,1.0 + self.ppo_eps)
        actor_loss = jnp.mean(jnp.maximum(cross_entropy1,cross_entropy2))
        entropy = prob * jnp.log(prob)
        entropy_loss = jnp.mean(entropy)
        total_loss = self.val_coef * critic_loss + actor_loss + ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)
    
    def _loss_continuous(self, params, obses, actions, vs, old_prob, adv, ent_coef, key):
        feature = self.preproc.apply(params, key, obses)
        vals = self.critic.apply(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(jax.lax.stop_gradient(vs) - vals))) #is sooooo large!!!
        #critic_loss =  jnp.mean(hubberloss(vs - vals, 1.0))
        
        prob = self.actor.apply(params, key, feature)
        prob, log_prob = self.get_logprob(prob, actions, key, out_prob=True)
        ratio = jnp.exp(log_prob - old_prob)
        cross_entropy1 = -adv*ratio; cross_entropy2 = -adv*jnp.clip(ratio,1.0 - self.ppo_eps,1.0 + self.ppo_eps)
        actor_loss = jnp.mean(jnp.maximum(cross_entropy1,cross_entropy2))
        mu, log_std = prob
        entropy_loss = - jnp.mean(0.5 + 0.5 * jnp.log(2 * np.pi) + log_std) #jnp.mean(jnp.square(mu) + jnp.square(log_std))
        total_loss = self.val_coef * critic_loss + actor_loss + ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)
    
    def learn(self, total_trainstep, callback=None, log_interval=10, tb_log_name="IMPALA_PPO",
                reset_num_timesteps=True, replay_wrapper=None):
        super().learn(total_trainstep, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)

from typing import List

@jax.jit
def convert_jax(obs : List):
  return [jax.device_get(o).astype(jnp.float32) for o in obs]