import jax
import jax.numpy as jnp
import optax

from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.math.jax_utils import convert_jax
from jax_baselines.math.returns import get_gaes, normalize_advantage


class SurrogatePolicyGradient(Actor_Critic_Policy_Gradient_Family):
    """Shared minibatch-epoch surrogate policy-gradient machinery for PPO and SPO.

    PPO and SPO share identical rollout preprocessing (GAE) and the
    minibatch/epoch optimization loop; they differ only in the per-sample actor
    loss supplied through ``_loss_discrete``/``_loss_continuous`` (wired to
    ``self._loss`` by ``Actor_Critic_Policy_Gradient_Family``). Subclasses keep
    their own ``__init__`` (name + defaults) and ``_loss_*`` implementations.
    """

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
        if self.gae_normalize and self.gae_normalize_scope == "batch":
            adv = normalize_advantage(adv)
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
                if self.gae_normalize and self.gae_normalize_scope == "minibatch":
                    adv = normalize_advantage(adv)
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
