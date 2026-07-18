import jax
import jax.numpy as jnp
import optax

from jax_baselines.IMPALA.base_class import IMPALA_Family
from jax_baselines.math.jax_utils import convert_jax


class SurrogateIMPALA(IMPALA_Family):
    """Shared V-trace + minibatch/epoch surrogate machinery for IMPALA_PPO/IMPALA_SPO.

    Both share identical model setup, V-trace preprocessing, and the
    minibatch/epoch optimization loop; they differ only in the per-sample actor
    loss supplied through ``_loss_discrete``/``_loss_continuous``. Subclasses
    keep those plus their own ``learn``.
    """

    def __init__(
        self,
        workers,
        model_builder_maker,
        runtime,
        ppo_eps=0.2,
        epoch_num=3,
        buffer_size=0,
        gamma=0.995,
        lamda=0.95,
        learning_rate=3e-4,
        update_freq=100,
        batch_size=1024,
        sample_size=1,
        val_coef=0.2,
        ent_coef=0.01,
        use_entropy_adv_shaping=True,
        entropy_adv_shaping_kappa=2.0,
        rho_max=1.0,
        log_interval=1,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        seed=None,
        optimizer_factory=None,
        worker_replay_factory=None,
        checkpoint_store=None,
    ):
        self.minibatch_size = 256
        self.epoch_num = epoch_num
        self.ppo_eps = ppo_eps

        super().__init__(
            workers,
            model_builder_maker,
            runtime=runtime,
            buffer_size=buffer_size,
            gamma=gamma,
            lamda=lamda,
            learning_rate=learning_rate,
            update_freq=update_freq,
            batch_size=batch_size,
            sample_size=sample_size,
            val_coef=val_coef,
            ent_coef=ent_coef,
            use_entropy_adv_shaping=use_entropy_adv_shaping,
            entropy_adv_shaping_kappa=entropy_adv_shaping_kappa,
            rho_max=rho_max,
            log_interval=log_interval,
            log_dir=log_dir,
            _init_setup_model=_init_setup_model,
            policy_kwargs=policy_kwargs,
            seed=seed,
            optimizer_factory=optimizer_factory,
            worker_replay_factory=worker_replay_factory,
            checkpoint_store=checkpoint_store,
        )

    def setup_model(self):
        self.model_builder = self.model_builder_maker(
            self.observation_space, self.action_size, self.action_type, self.policy_kwargs
        )
        self.actor_builder = self.get_actor_builder()

        self.preproc, self.actor, self.critic, self.params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.opt_state = self.optimizer.init(self.params)

        self._train_step = jax.jit(self._train_step)
        self.preprocess = jax.jit(self.preprocess)
        self._loss = (
            jax.jit(self._loss_discrete)
            if self.action_type == "discrete"
            else jax.jit(self._loss_continuous)
        )

    def train_step(self, steps):
        data = self.buffer.sample()

        (
            self.params,
            self.opt_state,
            critic_loss,
            actor_loss,
            entropy_loss,
            rho,
            targets,
        ) = self._train_step(
            self.params,
            self.opt_state,
            next(self.key_seq),
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
        )

        if steps % self.log_interval == 0:
            log_dict = {
                "loss/critic_loss": float(critic_loss),
                "loss/actor_loss": float(actor_loss),
                "loss/entropy_loss": float(entropy_loss),
                "loss/mean_rho": float(rho),
                "loss/mean_target": float(targets),
            }
            self.logger_server.log_trainer(steps, log_dict)
        return critic_loss, float(rho)

    def preprocess(
        self,
        params,
        key,
        obses,
        actions,
        mu_log_prob,
        rewards,
        nxtobses,
        terminateds,
        truncateds,
    ):
        # ((b x h x w x c), (b x n)) x worker -> (worker x b x h x w x c), (worker x b x n)
        obses = jax.tree.map(lambda *values: jnp.stack(values), *obses)
        nxtobses = jax.tree.map(lambda *values: jnp.stack(values), *nxtobses)
        actions = jnp.stack(actions)
        mu_log_prob = jnp.stack(mu_log_prob)
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
        vs, rho, adv = self._compute_vtrace(
            pi_prob, mu_log_prob, rewards, terminateds, truncateds, value, next_value
        )
        obses = {key: jnp.vstack(value) for key, value in obses.items()}
        actions = jnp.vstack(actions)
        vs = jnp.vstack(vs)
        mu_prob = jnp.vstack(mu_log_prob)
        pi_prob = jnp.vstack(pi_prob)
        rho = jnp.vstack(rho)
        adv = jnp.vstack(adv)
        return obses, actions, vs, mu_prob, pi_prob, rho, adv

    def _train_step(
        self,
        params,
        opt_state,
        key,
        obses,
        actions,
        mu_log_prob,
        rewards,
        nxtobses,
        terminateds,
        truncateds,
    ):
        obses, actions, vs, mu_prob, pi_prob, rho, adv = self.preprocess(
            params,
            key,
            obses,
            actions,
            mu_log_prob,
            rewards,
            nxtobses,
            terminateds,
            truncateds,
        )

        def i_f(idx, vals):
            params, opt_state, key, critic_loss, actor_loss, entropy_loss = vals
            use_key, key = jax.random.split(key)
            batch_idxes = jax.random.permutation(use_key, jnp.arange(vs.shape[0])).reshape(
                -1, self.minibatch_size
            )
            obses_batch = {key: value[batch_idxes] for key, value in obses.items()}
            actions_batch = actions[batch_idxes]
            vs_batch = vs[batch_idxes]
            mu_prob_batch = mu_prob[batch_idxes]
            pi_prob_batch = pi_prob[batch_idxes]
            adv_batch = adv[batch_idxes]

            def f(updates, input):
                params, opt_state, key = updates
                obs, act, vs, mu_prob, pi_prob, adv = input
                use_key, key = jax.random.split(key)
                (total_loss, (critic_loss, actor_loss, entropy_loss),), grad = jax.value_and_grad(
                    self._loss, has_aux=True
                )(params, obs, act, vs, mu_prob, pi_prob, adv, use_key)
                updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state, key), (critic_loss, actor_loss, entropy_loss)

            updates, losses = jax.lax.scan(
                f,
                (params, opt_state, key),
                (
                    obses_batch,
                    actions_batch,
                    vs_batch,
                    mu_prob_batch,
                    pi_prob_batch,
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
            jnp.mean(rho),
            jnp.mean(vs),
        )
