ENV_BACKENDS = ("gymnasium", "envpool", "mjlab")


def add_env_args(parser):
    parser.add_argument(
        "--env_backend",
        default="gymnasium",
        choices=ENV_BACKENDS,
        help="environment runtime backend",
    )
    parser.add_argument("--env_observation_key", help="dotted observation path")
    parser.add_argument("--env_episode_length", type=int, help="episode length in steps")
    parser.add_argument("--env_device", default="cuda:0", help="simulator device")
    parser.add_argument(
        "--env_jax_arrays", action="store_true", help="exchange mjlab tensors with JAX via DLPack"
    )
    parser.add_argument(
        "--env_reuse_for_eval",
        action="store_true",
        help="reuse the training environment for evaluation (Gymnasium or mjlab)",
    )


def env_builder_kwargs(args):
    return {
        "env_backend": args.env_backend,
        "observation_key": args.env_observation_key,
        "episode_length": args.env_episode_length,
        "device": args.env_device,
        "jax_arrays": args.env_jax_arrays,
        "reuse_for_eval": args.env_reuse_for_eval,
    }
