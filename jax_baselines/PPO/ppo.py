import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.common.utils import convert_jax, get_gaes


class PPO(Actor_Critic_Policy_Gradient_Family):
    def __init__(
        self,
        env_builder,
        model_builder_maker,
        lamda=0.95,
        gae_normalize=False,
        minibatch_size=32,
        epoch_num=4,
        ppo_eps=0.2,
        value_clip=0.5,
        **kwargs
    ):
        super().__init__(env_builder, model_builder_maker, **kwargs)

        self.name = "PPO"
        self.lamda = lamda
        self.gae_normalize = gae_normalize
        self.ppo_eps = ppo_eps
        self.value_clip = value_clip
        self.minibatch_size = minibatch_size
        self.batch_size = int(
            np.ceil(kwargs.get("batch_size", 256) * self.worker_size / minibatch_size)
            * minibatch_size
            / self.worker_size
        )
        self.epoch_num = epoch_num

        self.get_memory_setup()

        if kwargs.get("_init_setup_model", True):
            self.setup_model()

    def setup_model(self):
        self.model_builder = self.model_builder_maker(
            self.observation_space, self.action_size, self.action_type, self.policy_kwargs
        )

        self.preproc, self.actor, self.critic, self.params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.opt_state = self.optimizer.init(self.params)

        self._get_actions = jax.jit(self._get_actions)
        self._preprocess = jax.jit(self._preprocess)
        self._train_step = jax.jit(self._train_step)

    def train_step(self, steps):
        # Sample a batch from the replay buffer
        data = self.buffer.get_buffer()

        (
            self.params,
            self.opt_state,
            critic_loss,
            actor_loss,
            entropy_loss,
            targets,
        ) = self._train_step(self.params, self.opt_state, next(self.key_seq), **data)

        if self.logger_run:
            self.logger_run.log_metric("loss/critic_loss", critic_loss, steps)
            self.logger_run.log_metric("loss/actor_loss", actor_loss, steps)
            self.logger_run.log_metric("loss/entropy_loss", entropy_loss, steps)
            self.logger_run.log_metric("loss/mean_target", targets, steps)

        return critic_loss

    def _preprocess(self, params, key, obses, actions, rewards, nxtobses, terminateds, truncateds):
        obses = [jnp.stack(zo) for zo in zip(*obses)]
        nxtobses = [jnp.stack(zo) for zo in zip(*nxtobses)]
        actions = jnp.stack(actions)
        rewards = jnp.stack(rewards)
        terminateds = jnp.stack(terminateds)
        truncateds = jnp.stack(truncateds)
        obses = convert_jax(obses)
        nxtobses = convert_jax(nxtobses)
        feature = jax.vmap(self.preproc, in_axes=(None, None, 0))(params, key, obses)
        value = jax.vmap(self.critic, in_axes=(None, None, 0))(params, key, feature)
        next_value = jax.vmap(self.critic, in_axes=(None, None, 0))(
            params,
            key,
            jax.vmap(self.preproc, in_axes=(None, None, 0))(params, key, nxtobses),
        )
        pi_prob = jax.vmap(self.get_logprob, in_axes=(0, 0, None))(
            jax.vmap(self.actor, in_axes=(None, None, 0))(params, key, feature),
            actions,
            key,
        )
        adv = jax.vmap(get_gaes, in_axes=(0, 0, 0, 0, 0, None, None))(
            rewards, terminateds, truncateds, value, next_value, self.gamma, self.lamda
        )
        obses = [jnp.vstack(o) for o in obses]
        actions = jnp.vstack(actions)
        value = jnp.vstack(value)
        pi_prob = jnp.vstack(pi_prob)
        adv = jnp.vstack(adv)
        targets = value + adv
        if self.gae_normalize:
            adv = (adv - jnp.mean(adv, keepdims=True)) / (jnp.std(adv, keepdims=True) + 1e-6)
        return obses, actions, value, targets, pi_prob, adv

    def _train_step(
        self,
        params,
        opt_state,
        key,
        obses,
        actions,
        rewards,
        nxtobses,
        terminateds,
        truncateds,
    ):
        obses, actions, old_values, targets, act_prob, adv = self._preprocess(
            params, key, obses, actions, rewards, nxtobses, terminateds, truncateds
        )

        def i_f(idx, vals):
            params, opt_state, key, critic_loss, actor_loss, entropy_loss = vals
            use_key, key = jax.random.split(key)
            batch_idxes = jax.random.permutation(use_key, jnp.arange(targets.shape[0])).reshape(
                -1, self.minibatch_size
            )
            obses_batch = [o[batch_idxes] for o in obses]
            actions_batch = actions[batch_idxes]
            old_values_batch = old_values[batch_idxes]
            targets_batch = targets[batch_idxes]
            act_prob_batch = act_prob[batch_idxes]
            adv_batch = adv[batch_idxes]

            def f(updates, input):
                params, opt_state, key = updates
                obs, act, oldv, target, act_prob, adv = input
                use_key, key = jax.random.split(key)
                (total_loss, (c_loss, a_loss, entropy_loss)), grad = jax.value_and_grad(
                    self._loss, has_aux=True
                )(params, obs, act, oldv, target, act_prob, adv, use_key)
                updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state, key), (c_loss, a_loss, entropy_loss)

            updates, losses = jax.lax.scan(
                f,
                (params, opt_state, key),
                (
                    obses_batch,
                    actions_batch,
                    old_values_batch,
                    targets_batch,
                    act_prob_batch,
                    adv_batch,
                ),
            )
            params, opt_state, key = updates
            cl, al, el = losses
            critic_loss += jnp.mean(cl)
            actor_loss += jnp.mean(al)
            entropy_loss += jnp.mean(el)
            return params, opt_state, key, critic_loss, actor_loss, entropy_loss

        val = jax.lax.fori_loop(0, self.epoch_num, i_f, (params, opt_state, key, 0.0, 0.0, 0.0))
        params, opt_state, key, critic_loss, actor_loss, entropy_loss = val
        return (
            params,
            opt_state,
            critic_loss / self.epoch_num,
            actor_loss / self.epoch_num,
            entropy_loss / self.epoch_num,
            jnp.mean(targets),
        )

    def _loss_discrete(self, params, obses, actions, old_value, targets, old_prob, adv, key):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        vals_clip = old_value + jnp.clip(vals - old_value, -self.value_clip, self.value_clip)
        vf1 = jnp.square(vals - targets)
        vf2 = jnp.square(vals_clip - targets)
        critic_loss = jnp.mean(jnp.maximum(vf1, vf2))

        prob, log_prob = self.get_logprob(
            self.actor(params, key, feature), actions, key, out_prob=True
        )
        ratio = jnp.exp(log_prob - old_prob)
        cross_entropy1 = -adv * ratio
        cross_entropy2 = -adv * jnp.clip(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps)
        actor_loss = jnp.mean(jnp.maximum(cross_entropy1, cross_entropy2))
        # Numerical stability: avoid log(0) with small epsilon
        epsilon = 1e-8
        entropy_loss = jnp.mean(prob * jnp.log(jnp.maximum(prob, epsilon)))
        total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)

    def _loss_continuous(self, params, obses, actions, old_value, targets, old_prob, adv, key):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        vals_clip = old_value + jnp.clip(vals - old_value, -self.value_clip, self.value_clip)
        vf1 = jnp.square(jnp.squeeze(vals - targets))
        vf2 = jnp.square(jnp.squeeze(vals_clip - targets))
        critic_loss = jnp.mean(jnp.maximum(vf1, vf2))

        prob, log_prob = self.get_logprob(
            self.actor(params, key, feature), actions, key, out_prob=True
        )
        ratio = jnp.exp(log_prob - old_prob)
        cross_entropy1 = -adv * ratio
        cross_entropy2 = -adv * jnp.clip(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps)
        actor_loss = jnp.mean(jnp.maximum(cross_entropy1, cross_entropy2))
        mu, log_std = prob
        # Correct Gaussian entropy: -H = -sum(log_std) - 0.5*dim*(1+log(2π))
        dim = mu.shape[-1]
        neg_entropy = -jnp.sum(log_std, axis=-1) - 0.5 * dim * (1.0 + jnp.log(2.0 * jnp.pi))
        entropy_loss = jnp.mean(neg_entropy)
        total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="PPO",
        run_name="PPO",
    ):
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            experiment_name,
            run_name,
        )
