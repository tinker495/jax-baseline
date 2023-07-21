import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from copy import deepcopy

from haiku_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from haiku_baselines.TPPO.network import Actor, Critic
from haiku_baselines.common.Module import PreProcess
from haiku_baselines.common.utils import convert_jax, get_gaes, print_param, kl_divergence_discrete, kl_divergence_continuous

class TPPO(Actor_Critic_Policy_Gradient_Family):
	def __init__(self, env, gamma=0.995, lamda = 0.95, gae_normalize = False, learning_rate=3e-4, batch_size=256, minibatch_size=32, epoch_num=4,val_coef=0.5, ent_coef = 0.001, 
				 kl_range = 0.05, kl_coef = 5, log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
				 full_tensorboard_log=False, seed=None, optimizer = 'rmsprop'):
		
		super(TPPO, self).__init__(env, gamma, learning_rate, batch_size, val_coef, ent_coef,
				 log_interval, tensorboard_log, _init_setup_model, policy_kwargs, 
				 full_tensorboard_log, seed, optimizer)
		
		self.lamda = lamda
		self.gae_normalize = gae_normalize
		self.kl_range = kl_range
		self.kl_coef = kl_coef
		self.minibatch_size = minibatch_size
		self.batch_size = int(np.ceil(batch_size * self.worker_size / minibatch_size) * minibatch_size / self.worker_size)
		self.epoch_num = epoch_num
		
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
		self._preprocess = jax.jit(self._preprocess)
		self._train_step = jax.jit(self._train_step)
	
	def discription(self):
		return "score : {:.3f}, loss : {:.3f} |".format(
									np.mean(self.scoreque), np.mean(self.lossque)
									)

	
	def train_step(self, steps):
		# Sample a batch from the replay buffer
		data = self.buffer.get_buffer()
		
		self.params, self.opt_state, critic_loss, actor_loss, entropy_loss, kls, targets = self._train_step(self.params, self.opt_state, next(self.key_seq), **data)
		
		if self.summary:
			self.summary.add_scalar("loss/critic_loss", critic_loss, steps)
			self.summary.add_scalar("loss/actor_loss", actor_loss, steps)
			self.summary.add_scalar("loss/entropy_loss", entropy_loss, steps)
			self.summary.add_scalar("loss/mean_target", targets, steps)
			self.summary.add_scalar("loss/kl_divergense", kls, steps)
			
		return critic_loss
	
	def _preprocess(self, params, key, obses, actions, rewards, nxtobses, dones, terminals):
		obses = [jnp.stack(zo) for zo in zip(*obses)]; nxtobses = [jnp.stack(zo) for zo in zip(*nxtobses)]; actions = jnp.stack(actions)
		rewards = jnp.stack(rewards); dones = jnp.stack(dones); terminals = jnp.stack(terminals)
		obses = jax.vmap(convert_jax)(obses); nxtobses = jax.vmap(convert_jax)(nxtobses)
		feature = jax.vmap(self.preproc.apply, in_axes=(None,None,0))(params, key, obses)
		value = jax.vmap(self.critic.apply, in_axes=(None,None,0))(params, key, feature)
		next_value = jax.vmap(self.critic.apply, in_axes=(None,None,0))(params, key, jax.vmap(self.preproc.apply, in_axes=(None,None,0))(params, key, nxtobses))
		prob, pi_prob = jax.vmap(self.get_logprob, in_axes=(0,0,None,None))(jax.vmap(self.actor.apply,in_axes=(None,None,0))(params, key, feature), actions, key, True)
		adv = jax.vmap(get_gaes, in_axes=(0,0,0,0,0,None,None))(rewards,dones,terminals,value,next_value,self.gamma,self.lamda)
		obses = [jnp.vstack(o) for o in obses]; actions = jnp.vstack(actions); value = jnp.vstack(value); prob = jnp.vstack(prob); pi_prob = jnp.vstack(pi_prob)
		adv = jnp.vstack(adv); targets = value + adv
		if self.gae_normalize:
			adv = (adv - jnp.mean(adv,keepdims=True)) / (jnp.std(adv,keepdims=True) + 1e-6)
		return obses, actions, targets, prob, pi_prob, adv
	
	def _train_step(self, params, opt_state, key, obses, actions, rewards, nxtobses, dones, terminals):
		obses, actions, targets, old_prob, old_act_prob, adv = self._preprocess(params, key, obses, actions, rewards, nxtobses, dones, terminals)
		def i_f(idx, vals):
			params, opt_state, key, critic_loss, actor_loss, entropy_loss, kls = vals
			use_key, key = jax.random.split(key)
			batch_idxes = jax.random.permutation(use_key, jnp.arange(targets.shape[0])).reshape(-1,self.minibatch_size)
			obses_batch = [o[batch_idxes] for o in obses]
			actions_batch = actions[batch_idxes]
			targets_batch = targets[batch_idxes]
			old_prob_batch = old_prob[batch_idxes]
			old_act_prob_batch = old_act_prob[batch_idxes]
			adv_batch = adv[batch_idxes]
			def f(updates, input):
				params, opt_state, key = updates
				obs, act, target, old_prob, old_act_prob, adv = input
				use_key, key = jax.random.split(key)
				(total_loss, (c_loss, a_loss, entropy_loss, kl)), grad = \
					jax.value_and_grad(self._loss,has_aux = True)(params, obs, act, target, old_prob, old_act_prob, adv, use_key)
				updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
				params = optax.apply_updates(params, updates)
				return (params, opt_state, key), (c_loss, a_loss, entropy_loss, kl)
			updates, losses  = jax.lax.scan(f, (params, opt_state, key), (obses_batch, actions_batch, targets_batch, old_prob_batch, old_act_prob_batch, adv_batch))
			params, opt_state, key = updates
			cl, al, el, kl = losses
			critic_loss += jnp.mean(cl); actor_loss += jnp.mean(al); entropy_loss += jnp.mean(el); kls += jnp.mean(kl)
			return params, opt_state, key, critic_loss, actor_loss, entropy_loss, kls
		val = jax.lax.fori_loop(0, self.epoch_num, i_f, (params, opt_state, key, 0.0, 0.0, 0.0, 0.0))
		params, opt_state, key, critic_loss, actor_loss, entropy_loss, kls = val
		return params, opt_state, critic_loss/self.epoch_num, actor_loss/self.epoch_num, entropy_loss/self.epoch_num, kls/self.epoch_num, jnp.mean(targets)

	def _loss_discrete(self, params, obses, actions, targets, old_prob, old_act_prob, adv, key):
		feature = self.preproc.apply(params, key, obses)
		vals = self.critic.apply(params, key, feature)
		critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))
		
		prob, log_prob = self.get_logprob(self.actor.apply(params, key, feature), actions, key, out_prob=True)
		ratio = jnp.exp(log_prob - old_act_prob)
		kl = jax.vmap(kl_divergence_discrete)(old_prob, prob)
		actor_loss = - jnp.mean(jnp.where((kl >= self.kl_range) & (ratio > 1.0), adv * ratio - self.kl_coef * kl, adv * ratio))
		entropy = prob * jnp.log(prob)
		entropy_loss = jnp.mean(entropy)
		total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
		return total_loss, (critic_loss, actor_loss, entropy_loss, jnp.mean(kl))
	
	def _loss_continuous(self, params, obses, actions, targets, old_prob, old_act_prob, adv, key):
		feature = self.preproc.apply(params, key, obses)
		vals = self.critic.apply(params, key, feature)
		critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))
		
		prob, log_prob = self.get_logprob(self.actor.apply(params, key, feature), actions, key, out_prob=True)
		ratio = jnp.exp(log_prob - old_act_prob)
		kl = jax.vmap(kl_divergence_continuous)(old_prob, prob)
		actor_loss = - jnp.mean(jnp.where((kl >= self.kl_range) & (ratio > 1.0), adv * ratio - self.kl_coef * kl, adv * ratio))
		mu, log_std = prob
		entropy_loss = jnp.mean(jnp.square(mu) - log_std)
		total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
		return total_loss, (critic_loss, actor_loss, entropy_loss, jnp.mean(kl))
	
	def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="TPPO",
			  reset_num_timesteps=True, replay_wrapper=None):
		
		super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)