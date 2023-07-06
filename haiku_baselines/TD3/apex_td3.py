import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from copy import deepcopy
from itertools import repeat

from haiku_baselines.APE_X.dpg_base_class import Ape_X_Deteministic_Policy_Gradient_Family
from haiku_baselines.TD3.network import Actor, Critic
from haiku_baselines.common.Module import PreProcess
from haiku_baselines.common.utils import soft_update, convert_jax, print_param

class APE_X_TD3(Ape_X_Deteministic_Policy_Gradient_Family):

	def __init__(self, workers, manager = None, gamma=0.995, learning_rate=5e-5, buffer_size=50000, exploration_initial_eps=0.9, exploration_decay=0.7, 
					target_action_noise_mul = 1.5, batch_size=32, policy_delay = 3,
					n_step = 1, learning_starts=1000, target_network_update_tau=5e-4, gradient_steps = 1, 
					prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_eps=1e-3,
					log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
					full_tensorboard_log=False, seed=None, optimizer = 'adamw', compress_memory = False):
		super().__init__(workers, manager, gamma, learning_rate, buffer_size, exploration_initial_eps, exploration_decay, batch_size,
					n_step, learning_starts, target_network_update_tau, gradient_steps, 
					prioritized_replay_alpha, prioritized_replay_beta0, prioritized_replay_eps,
					log_interval, tensorboard_log, _init_setup_model, policy_kwargs, 
					full_tensorboard_log, seed, optimizer, compress_memory)
		
		self.action_noise = self.exploration_initial_eps**(1 + self.exploration_decay)
		self.target_action_noise = self.action_noise*target_action_noise_mul
		self.action_noise_clamp = 0.5 #self.target_action_noise*1.5
		self.policy_delay = policy_delay
		
		if _init_setup_model:
			self.setup_model()

	def setup_model(self):
		self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
		if 'cnn_mode' in self.policy_kwargs.keys():
			cnn_mode = self.policy_kwargs['cnn_mode']
			del self.policy_kwargs['cnn_mode']
			
		def network_builder(observation_space, cnn_mode, action_size, **kwargs):
			def builder():
				preproc = hk.transform(lambda x: PreProcess(observation_space, cnn_mode=cnn_mode)(x))
				actor = hk.transform(lambda x: Actor(action_size,
								**kwargs)(x))
				critic = hk.transform(lambda x,a: (Critic(**kwargs)(x,a),Critic(**kwargs)(x,a)))
				return preproc, actor, critic
			return builder
		self.network_builder = network_builder(self.observation_space, cnn_mode, self.action_size, **self.policy_kwargs)
		self.actor_builder = self.get_actor_builder()

		self.preproc, self.actor, self.critic = self.network_builder()

		pre_param = self.preproc.init(next(self.key_seq),
							[np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
		feature = self.preproc.apply(pre_param, None, [np.zeros((1,*o),dtype=np.float32) for o in self.observation_space])
		actor_param = self.actor.init(next(self.key_seq), feature)
		critic_param = self.critic.init(next(self.key_seq), feature, np.zeros((1,self.action_size[0])))
		self.params = hk.data_structures.merge(pre_param, actor_param, critic_param)
		self.target_params = deepcopy(self.params)
		
		self.opt_state = self.optimizer.init(self.params)
		
		print("----------------------model----------------------")
		print_param('preprocess',pre_param)
		print_param('actor',actor_param)
		print_param('critic',critic_param)
		print("-------------------------------------------------")

		self._get_actions = jax.jit(self._get_actions)
		self._train_step = jax.jit(self._train_step)
		
	def get_actor_builder(self):
		gamma = self._gamma
		action_size = self.action_size[0]
		prioritized_replay_eps = self.prioritized_replay_eps
		action_noise_clamp = self.action_noise_clamp
		target_action_noise = self.target_action_noise
		class Noise():
			def __init__(self, action_size) -> None:
				self.action_size = action_size

			def __call__(self):
				return np.random.normal(size=(1,self.action_size))
			
			def reset(self, worker_id):
				pass

		def builder():
			noise = Noise(action_size)
			key_seq = hk.PRNGSequence(42)

			def get_abs_td_error(actor, critic, preproc, params, obses, actions, rewards, nxtobses, dones, key):
				size = obses[0].shape[0]
				next_feature = preproc.apply(params, key, convert_jax(nxtobses))
				next_action = jnp.clip(
					  actor.apply(params, key, next_feature) \
					  + jnp.clip(target_action_noise*jax.random.normal(key,(size,action_size)),-action_noise_clamp,action_noise_clamp)
					  ,-1.0,1.0)
				q1, q2 = critic.apply(params, key, next_feature, next_action)
				next_q = jnp.minimum(q1,q2)
				feature = preproc.apply(params, key, convert_jax(obses))
				q_values1, q_values2 = critic.apply(params, key, feature, actions)
				target = rewards + gamma * (1.0 - dones) * next_q
				td1_error = jnp.abs(q_values1 - target)
				return jnp.squeeze(td1_error)

			def actor(actor, preproc, params, obses, key):
				return actor.apply(params, key, preproc.apply(params, key, convert_jax(obses)))

			def get_action(actor, params, obs, noise, epsilon, key):
				actions = np.clip(np.asarray(actor(params ,obs, key)) + noise()*epsilon,-1,1)[0]
				return actions

			def random_action(params, obs, noise, epsilon, key):
				return np.random.uniform(-1.0,1.0,size=(action_size))
			
			return get_abs_td_error, actor, get_action, random_action, noise, key_seq
		return builder
	
	def train_step(self, steps, gradient_steps):
		# Sample a batch from the replay buffer
		for _ in range(gradient_steps):
			data = self.replay_buffer.sample(self.batch_size,self.prioritized_replay_beta0)        
			
			self.params, self.target_params, self.opt_state, loss, t_mean, new_priorities = \
				self._train_step(self.params, self.target_params, self.opt_state, next(self.key_seq), steps, **data)
			
			self.replay_buffer.update_priorities(data['indexes'], new_priorities)
			
		if steps % self.log_interval == 0:
			log_dict = {"loss/qloss": float(loss), "loss/targets": float(t_mean)}
			self.logger_server.log_trainer.remote(steps, log_dict)
			
		return loss

	def _train_step(self, params, target_params, opt_state, key, step,
					obses, actions, rewards, nxtobses, dones, weights=1, indexes=None):
		obses = convert_jax(obses); nxtobses = convert_jax(nxtobses); not_dones = 1.0 - dones
		targets = self._target(target_params, rewards, nxtobses, not_dones, key)
		(total_loss, (critic_loss, actor_loss, abs_error)), grad = jax.value_and_grad(self._loss,has_aux = True)(params, obses, actions, targets, weights, key, step)
		updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
		params = optax.apply_updates(params, updates)
		target_params = soft_update(params, target_params, self.target_network_update_tau)
		new_priorities = abs_error
		return params, target_params, opt_state, critic_loss, -actor_loss, new_priorities
	
	def _loss(self, params, obses, actions, targets, weights, key, step):
		feature = self.preproc.apply(params, key, obses)
		q1, q2 = self.critic.apply(params, key, feature, actions)
		error1 = jnp.squeeze(q1 - targets)
		error2 = jnp.squeeze(q2 - targets)
		critic_loss = jnp.mean(weights*jnp.square(error1)) + jnp.mean(weights*jnp.square(error2))
		policy = self.actor.apply(params, key, feature)
		vals, _ = self.critic.apply(jax.lax.stop_gradient(params), key, feature, policy)
		actor_loss = jnp.mean(-vals)
		total_loss = jax.lax.select(step % self.policy_delay == 0, critic_loss + actor_loss, critic_loss)
		return total_loss, (critic_loss, actor_loss, jnp.abs(error1))
	
	def _target(self, target_params, rewards, nxtobses, not_dones, key):
		next_feature = self.preproc.apply(target_params, key, nxtobses)
		next_action = jnp.clip(
					  self.actor.apply(target_params, key, next_feature) \
					  + jnp.clip(self.target_action_noise*jax.random.normal(key,(self.batch_size,self.action_size[0])),-self.action_noise_clamp,self.action_noise_clamp)
					  ,-1.0,1.0)
		q1, q2 = self.critic.apply(target_params, key, next_feature, next_action)
		next_q = jnp.minimum(q1,q2)
		return (not_dones * next_q * self._gamma) + rewards

	
	def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="Ape_X_TD3",
			  reset_num_timesteps=True, replay_wrapper=None):
		super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)