from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Protocol


@dataclass(frozen=True)
class AlgoSpec:
    cls: type | Callable[
        [Namespace], type
    ]  # agent class, or args -> class (e.g. hl_gauss variants)
    builder: str  # maker filename token, e.g. "ddpg"
    build: Callable[[Namespace], dict]  # args -> constructor kwargs
    # (must NOT include env_builder / model_builder_maker / policy_kwargs;
    #  those are threaded by run_family)

    def resolve_cls(self, args: Namespace) -> type:
        return self.cls if isinstance(self.cls, type) else self.cls(args)


@dataclass(frozen=True)
class FamilyRunner:
    add_args: Callable[[ArgumentParser], None]
    build_env: Callable[[Namespace], tuple]  # args -> (env_builder, policy_kwargs)
    algos: dict[str, AlgoSpec]
    maker_pkg: str  # builder package with a "{lib}" placeholder, e.g. "model_builder.{lib}.dpg"
    variant: Callable[[Namespace], str]  # args -> "" | "simba_" | "simbav2_"


class _MakerRunner(Protocol):
    """Structural interface `resolve_maker` needs from any family runner."""

    maker_pkg: str
    variant: Callable[[Namespace], str]


def resolve_maker(runner: _MakerRunner, spec: AlgoSpec, args: Namespace):
    module = f"{runner.maker_pkg.format(lib=args.model_lib)}.{runner.variant(args)}{spec.builder}_builder"
    try:
        mod = import_module(module)
    except ModuleNotFoundError as exc:
        # Convert to a clean "unsupported combo" only when the target builder
        # module itself is missing (an unsupported algo/lib/variant). If a
        # different module is missing, the builder exists but a transitive
        # import failed, so re-raise to surface the real bug.
        if exc.name is None or not (module == exc.name or module.startswith(exc.name + ".")):
            raise
        raise SystemExit(
            f"unsupported combo: algo={args.algo} model_lib={args.model_lib} "
            f"variant={runner.variant(args) or 'base'} (no module {module})"
        ) from exc
    return mod.model_builder_maker


def run_family(runner: FamilyRunner, argv=None):
    parser = ArgumentParser()
    runner.add_args(parser)
    args = parser.parse_args(argv)
    if args.algo not in runner.algos:
        raise SystemExit(f"unknown algo '{args.algo}', expected one of {sorted(runner.algos)}")
    spec = runner.algos[args.algo]
    maker = resolve_maker(runner, spec, args)
    env_builder, policy_kwargs = runner.build_env(args)
    agent = spec.resolve_cls(args)(
        env_builder, maker, policy_kwargs=policy_kwargs, **spec.build(args)
    )
    agent.learn(
        int(args.steps),
        experiment_name=args.experiment_name,
        eval_num=args.eval_num,
    )
    agent.test()
    return agent


@dataclass(frozen=True)
class DistributedFamilyRunner:
    """Sibling of FamilyRunner for Ray-based families (APE-X / IMPALA).

    The agent is built as ``cls(workers, maker, manager, **build)`` (not from an
    env_builder), needs a Ray runtime plus an mp.Manager and a list of worker
    actors, and runs ``learn(steps)`` only (no experiment_name, no test()).
    """

    add_args: Callable[[ArgumentParser], None]
    make_workers: Callable[[Namespace], list]  # args -> list of Ray worker actors
    policy_kwargs: Callable[[Namespace], dict]
    algos: dict[str, AlgoSpec]
    maker_pkg: str  # builder package with a "{lib}" placeholder
    variant: Callable[[Namespace], str]
    ray_cpu_headroom: int  # extra CPUs reserved beyond --worker


def run_distributed_family(runner: DistributedFamilyRunner, argv=None):
    import multiprocessing as mp

    import ray

    parser = ArgumentParser()
    runner.add_args(parser)
    args = parser.parse_args(argv)
    if args.algo not in runner.algos:
        raise SystemExit(f"unknown algo '{args.algo}', expected one of {sorted(runner.algos)}")
    spec = runner.algos[args.algo]
    maker = resolve_maker(runner, spec, args)
    manager = mp.get_context().Manager()
    ray.init(num_cpus=args.worker + runner.ray_cpu_headroom, num_gpus=0)
    workers = runner.make_workers(args)
    agent = spec.resolve_cls(args)(
        workers, maker, manager, policy_kwargs=runner.policy_kwargs(args), **spec.build(args)
    )
    agent.learn(int(args.steps))
    return agent
