import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from copy import deepcopy
from itertools import repeat

from haiku_baselines.APE_X.base_class import Ape_X_Family
from haiku_baselines.QRDQN.network import Model
from haiku_baselines.common.Module import PreProcess

from haiku_baselines.common.utils import hard_update, convert_jax, print_param, q_log_pi
from haiku_baselines.common.losses import QuantileHuberLosses

class APE_X_QRDQN(Ape_X_Family):

    def __init__(self, workers, gamma=0.995, learning_rate=5e-5, buffer_size=50000, exploration_initial_eps=0.8, exploration_decay=0.7, batch_size=32, double_q=False,
                 dueling_model = False, n_step = 1, learning_starts=1000, target_network_update_freq=2000, gradient_steps = 1,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_eps=1e-3, 
                 param_noise=False, munchausen=False, log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 n_support = 200, delta = 0.1,
                 full_tensorboard_log=False, seed=None, optimizer = 'adamw', compress_memory = False):
        super().__init__(workers, gamma, learning_rate, buffer_size, exploration_initial_eps, exploration_decay, batch_size, double_q,
                    dueling_model, n_step, learning_starts, target_network_update_freq, gradient_steps,
                    prioritized_replay_alpha, prioritized_replay_beta0, prioritized_replay_eps, 
                    param_noise, munchausen, log_interval, tensorboard_log, _init_setup_model, policy_kwargs, 
                    full_tensorboard_log, seed, optimizer, compress_memory)
        
        self.n_support = n_support
        self.delta = delta

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
        if 'cnn_mode' in self.policy_kwargs.keys():
            cnn_mode = self.policy_kwargs['cnn_mode']
            del self.policy_kwargs['cnn_mode']
            
        def network_builder(observation_space, cnn_mode, action_size, dueling_model, param_noise, support_n, **kwargs):
            def builder():
                preproc = hk.transform(lambda x: PreProcess(observation_space, cnn_mode=cnn_mode)(x))
                model = hk.transform(lambda x: Model(action_size,
                                dueling=dueling_model,noisy=param_noise, support_n=support_n,
                                **kwargs)(x))
                return preproc, model
            return builder
        self.network_builder = network_builder(self.observation_space, cnn_mode, self.action_size, self.dueling_model, self.param_noise, self.n_support, **self.policy_kwargs)

        self.preproc, self.model = self.network_builder()
        pre_param = self.preproc.init(next(self.key_seq),
                            [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
        model_param = self.model.init(next(self.key_seq),
                            self.preproc.apply(pre_param, 
                            None, [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space]))
        self.params = hk.data_structures.merge(pre_param, model_param)
        self.target_params = deepcopy(self.params)
        
        self.opt_state = self.optimizer.init(self.params)

        self.quantile = (jnp.linspace(0.0,1.0,self.n_support+1)[1:] + jnp.linspace(0.0,1.0,self.n_support+1)[:-1])/2.0   # [support]
        self.quantile = jax.device_put(jnp.expand_dims(self.quantile,axis=(0,1)))                                        # [1 x 1 x support]

        self.actor_builder = self.get_actor_builder()
        
        print("----------------------model----------------------")
        print_param('preprocess',pre_param)
        print_param('model',model_param)
        print("loss : quaile_huber_loss")
        print("-------------------------------------------------")

        self.get_q = jax.jit(self.get_q)
        self._loss = jax.jit(self._loss)
        self._target = jax.jit(self._target)
        self._train_step = jax.jit(self._train_step)
    
    def get_q(self, params, obses, key = None) -> jnp.ndarray:
        return self.model.apply(params, key, self.preproc.apply(params, key, obses))
        
    def get_actor_builder(self):
        gamma = self._gamma
        action_size = self.action_size[0]
        param_noise = self.param_noise
        quantile = self.quantile
        delta = self.delta
        prioritized_replay_eps = self.prioritized_replay_eps
        def builder():
            if param_noise:
                key_seq = hk.PRNGSequence(42)
            else:
                #make repeat None
                key_seq = repeat(None)

            def get_abs_td_error(model, preproc, params, obses, actions, rewards, nxtobses, dones, key):
                q_values = jnp.take_along_axis(model.apply(params, key, preproc.apply(params, key, convert_jax(obses))), jnp.expand_dims(actions.astype(jnp.int32),axis=2), axis=1)
                next_q = model.apply(params, key, preproc.apply(params, key, convert_jax(nxtobses)))
                next_actions = jnp.expand_dims(jnp.argmax(jnp.mean(next_q,axis=2),axis=1),axis=(1,2))
                next_vals = jnp.squeeze(jnp.take_along_axis(next_q, next_actions, axis=1)) # batch x support
                target = rewards + gamma * (1.0 - dones) * next_vals
                loss = QuantileHuberLosses(q_values, jnp.expand_dims(target,axis=2), quantile, delta)
                return jnp.squeeze(loss + prioritized_replay_eps)

            def actor(model, preproc, params, obses, key):
                q_values = model.apply(params, key, preproc.apply(params, key, convert_jax(obses)))
                return jnp.expand_dims(jnp.argmax(jnp.mean(q_values,axis=2),axis=1),axis=1)
            
            if param_noise:
                def get_action(actor, params, obs, epsilon, key):
                    return int(np.asarray(actor(params, obs, key))[0])
            else:
                def get_action(actor, params, obs, epsilon, key):
                    if epsilon <= np.random.uniform(0,1):
                        actions = int(np.asarray(actor(params, obs, key))[0])
                    else:
                        actions = np.random.choice(action_size)
                    return actions
                
            def random_action(params, obs, epsilon, key):
                return np.random.choice(action_size)

            return get_abs_td_error, actor, get_action, random_action, key_seq
        return builder
    
    def train_step(self, steps, gradient_steps):
        # Sample a batch from the replay buffer
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(self.batch_size,self.prioritized_replay_beta0)
            
            self.params, self.target_params, self.opt_state, loss, t_mean, new_priorities = \
                self._train_step(self.params, self.target_params, self.opt_state, steps, 
                                 next(self.key_seq) if self.param_noise else None,**data)
            
            self.replay_buffer.update_priorities(data['indexes'], new_priorities)
            
        if steps % self.log_interval == 0:
            log_dict = {"loss/qloss": float(loss), "loss/targets": float(t_mean)}
            self.logger_server.log_trainer.remote(steps, log_dict)
            
        return loss

    def _train_step(self, params, target_params, opt_state, steps, key, 
                    obses, actions, rewards, nxtobses, dones, weights=1, indexes=None):
        obses = convert_jax(obses); nxtobses = convert_jax(nxtobses); actions = jnp.expand_dims(actions.astype(jnp.int32),axis=2); not_dones = 1.0 - dones
        targets = self._target(params, target_params, obses, actions, rewards, nxtobses, not_dones, key)
        (loss,abs_error), grad = jax.value_and_grad(self._loss,has_aux = True)(params, obses, actions, targets, weights, key)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        target_params = hard_update(params, target_params, steps, self.target_network_update_freq)
        new_priorities = abs_error
        return params, target_params, opt_state, loss, jnp.mean(targets), new_priorities
    
    def _loss(self, params, obses, actions, targets, weights, key):
        theta_loss_tile = jnp.take_along_axis(self.get_q(params, obses, key), actions, axis=1)  # batch x 1 x support
        logit_valid_tile = jnp.expand_dims(targets,axis=2)                                      # batch x support x 1
        loss = QuantileHuberLosses(theta_loss_tile, logit_valid_tile, self.quantile, self.delta)
        return jnp.mean(loss * weights), loss #remove weight multiply cpprb weight is something wrong
    
    def _target(self,params, target_params, obses, actions, rewards, nxtobses, not_dones, key):
        next_q = self.get_q(target_params,nxtobses,key)
            
        if self.munchausen:
            if self.double_q:
                next_q_mean = jnp.mean(self.get_q(params,nxtobses,key),axis=2)
            else:
                next_q_mean = jnp.mean(next_q,axis=2)
            next_sub_q, tau_log_pi_next = q_log_pi(next_q_mean, self.munchausen_entropy_tau)
            pi_next = jax.nn.softmax(next_sub_q/self.munchausen_entropy_tau,axis=1)
            p_cuml = jnp.expand_dims(jnp.cumsum(pi_next,axis=1),axis=2).tile(32)
            r = jax.random.uniform(key, (32,1,32), dtype=p_cuml.dtype)
            ind = jnp.swapaxes(jax.vmap(jax.vmap(lambda p,r: jnp.searchsorted(p, r),in_axes=(1,1)))(p_cuml,r),1,2)
            sampled_q = jnp.take_along_axis(next_q - jnp.expand_dims(tau_log_pi_next,axis=2),ind,axis=1).squeeze()
            next_vals = sampled_q * not_dones
            
            if self.double_q:
                q_k_targets = jnp.mean(self.get_q(params,obses,key),axis=2)
            else:
                q_k_targets = jnp.mean(self.get_q(target_params,obses,key),axis=2)
            q_k_targets = jnp.mean(self.get_q(target_params,obses,key),axis=2)
            q_sub_targets, tau_log_pi = q_log_pi(q_k_targets, self.munchausen_entropy_tau)
            log_pi = q_sub_targets - self.munchausen_entropy_tau*tau_log_pi
            munchausen_addon = jnp.take_along_axis(log_pi,jnp.squeeze(actions,axis=2),axis=1)
            
            rewards = rewards + self.munchausen_alpha*jnp.clip(munchausen_addon, a_min=-1, a_max=0)
        else:
            if self.double_q:
                next_actions = jnp.expand_dims(jnp.argmax(jnp.mean(self.get_q(params,nxtobses,key),axis=2),axis=1),axis=(1,2))
            else:
                next_actions = jnp.expand_dims(jnp.argmax(jnp.mean(next_q,axis=2),axis=1),axis=(1,2))
            next_vals = not_dones * jnp.squeeze(jnp.take_along_axis(next_q, next_actions, axis=1)) # batch x support
        return (next_vals * self._gamma) + rewards                                                 # batch x support

    
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="Ape_X_QRDQN",
              reset_num_timesteps=True, replay_wrapper=None):
        super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)