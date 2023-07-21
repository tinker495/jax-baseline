import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from copy import deepcopy

from haiku_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from haiku_baselines.TD3.network import Actor, Critic
from haiku_baselines.common.Module import PreProcess
from haiku_baselines.common.utils import soft_update, convert_jax, print_param

class TD3(Deteministic_Policy_Gradient_Family):
	def __init__(self, env, gamma=0.995, learning_rate=3e-4, buffer_size=100000,target_action_noise_mul = 1.5, 
				 action_noise = 0.1, train_freq=1, gradient_steps=1, batch_size=32, policy_delay = 3,
				 n_step = 1, learning_starts=1000, target_network_update_tau=5e-4, prioritized_replay=False,
				 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_eps=1e-3, 
				 log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
				 full_tensorboard_log=False, seed=None, optimizer = 'adamw'):
		
		super(TD3, self).__init__(env, gamma, learning_rate, buffer_size, train_freq, gradient_steps, batch_size,
				 n_step, learning_starts, target_network_update_tau, prioritized_replay,
				 prioritized_replay_alpha, prioritized_replay_beta0, prioritized_replay_eps,
				 log_interval, tensorboard_log, _init_setup_model, policy_kwargs, 
				 full_tensorboard_log, seed, optimizer)
		
		self.name = "TD3"
		self.action_noise = action_noise
		self.target_action_noise = action_noise*target_action_noise_mul
		self.action_noise_clamp = 0.5 #self.target_action_noise*1.5
		self.policy_delay = policy_delay
		
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
		self.critic = hk.transform(lambda x,a: (Critic(**self.policy_kwargs)(x,a), Critic(**self.policy_kwargs)(x,a)))
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
		
	def _get_actions(self, params, obses, key = None) -> jnp.ndarray:
		return self.actor.apply(params, key, self.preproc.apply(params, key, convert_jax(obses))) #
	
	def discription(self):
		return "score : {:.3f}, loss : {:.3f} |".format(
									np.mean(self.scoreque), np.mean(self.lossque)
									)
	
	def actions(self,obs,steps):
		if self.learning_starts < steps:
			actions = np.clip(np.asarray(self._get_actions(self.params,obs, None)) + 
							np.random.normal(0,self.action_noise,size=(self.worker_size,self.action_size[0])),-1,1)
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
	
	def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="TD3",
			  reset_num_timesteps=True, replay_wrapper=None):
		super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)