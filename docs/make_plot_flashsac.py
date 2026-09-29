"""FlashSAC vs the PyTorch reference: figures and tables for docs/flash_sac_implement.md.

Run from the repository root:  uv run python docs/make_plot_flashsac.py
Reads docs/csv/flashsac/. For every environment with finished runs it redraws the figure under
docs/figures/, rewrites the matching generated block in docs/flash_sac_implement.md, and prints it.
Curves are seed means with a min–max band; table cells are mean ± sample std over seeds.
"""

import csv
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

CSV_DIR = Path("docs/csv/flashsac")
DOC = Path("docs/flash_sac_implement.md")
IMPLS = {
    "upstream": ("Upstream FlashSAC (PyTorch)", "#2a78d6"),
    "jax": ("jax-baseline FlashSAC (JAX)", "#eb6834"),
}


@dataclass(frozen=True)
class EnvSpec:
    name: str
    title: str
    figure: str
    final: str
    episodes: int
    digits: int
    seeds: dict[str, int]  # planned seeds per implementation


ENVS = {
    "humanoid": EnvSpec(
        name="Humanoid-v4",
        title="Humanoid-v4 · 1 env (CPU simulation)",
        figure="docs/figures/flashsac_humanoid_v4_1m.png",
        final="1M",
        episodes=50,
        digits=0,
        # Upstream Humanoid runs one seed (2 h each); JAX runs five.
        seeds={"upstream": 1, "jax": 5},
    ),
    "mjlab": EnvSpec(
        name="mjlab G1 flat velocity",
        title="mjlab Unitree G1 flat velocity · 1024 envs (GPU simulation)",
        figure="docs/figures/flashsac_mjlab_g1_50m.png",
        final="50M",
        episodes=1024,
        digits=1,
        seeds={"upstream": 5, "jax": 5},
    ),
}
INK, MUTED, SURFACE, BAND = "#0b0b0b", "#52514e", "#fcfcfb", "#b8b7b1"


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load(env):
    """impl -> seed -> (eval curve [(step, hours, value)], wall-clock hours, throughput)."""
    curves = defaultdict(list)
    for row in read(CSV_DIR / f"{env}_eval.csv"):
        curves[row["impl"], int(row["seed"])].append(
            (int(row["step"]), float(row["hours"]), float(row["value"]))
        )
    runs = defaultdict(dict)
    for row in read(CSV_DIR / f"{env}_runs.csv"):
        key = row["impl"], int(row["seed"])
        runs[key[0]][key[1]] = (
            sorted(curves[key]),
            float(row["hours"]),
            float(row["throughput"]),
        )
    return runs


def paper_humanoid():
    by_step = defaultdict(list)
    for row in read(CSV_DIR / "humanoid_paper.csv"):
        by_step[int(row["step"])].append(float(row["value"]))
    return sorted(by_step.items())


def stat(values, digits, unit=""):
    text = f"{statistics.mean(values):,.{digits}f}"
    if len(values) > 1:
        text += f" ± {statistics.stdev(values):,.{digits}f}"
    return text + unit


def table(env, runs, threshold, threshold_label):
    cfg, digits = ENVS[env], ENVS[env].digits
    rows = [
        [
            "Implementation",
            "Seeds",
            f"Wall-clock to {cfg.final} steps",
            "Train throughput (env steps/s)",
            f"Time to {threshold:,.{digits}f} ({threshold_label})",
            f"Final eval @{cfg.final}",
            "Best eval",
        ]
    ]
    for impl, (label, _) in IMPLS.items():
        seeds = runs.get(impl, {})
        if not seeds:
            rows.append([label, f"0/{cfg.seeds[impl]}", *["pending"] * 5])
            continue
        curves = [curve for curve, _, _ in seeds.values()]
        reached = [next((h for _, h, v in c if v >= threshold), None) for c in curves]
        reached = [h for h in reached if h is not None]
        finals = [c[-1][2] for c in curves]
        time_to = "not reached" if not reached else stat(reached, 2, " h")
        if reached and len(reached) < len(curves):
            time_to += f" ({len(reached)}/{len(curves)} seeds)"
        final = stat(finals, digits)
        if len(finals) > 1:
            final += f" ({min(finals):,.{digits}f}–{max(finals):,.{digits}f})"
        rows.append(
            [
                label,
                f"{len(seeds)}/{cfg.seeds[impl]}",
                stat([h for _, h, _ in seeds.values()], 2, " h"),
                stat([t for _, _, t in seeds.values()], 0),
                time_to,
                final,
                stat([max(v for _, _, v in c) for c in curves], digits),
            ]
        )
    if env == "humanoid":
        finals = paper_humanoid()[-1][1]
        paper = f"{statistics.mean(finals):,.0f} ({min(finals):,.0f}–{max(finals):,.0f})"
        rows.append([f"Paper ({len(finals)} seeds)", "–", "–", "–", "–", paper, "–"])
    # Pad cells to column width like mdformat, the repo's pre-commit Markdown formatter.
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in rows]
    lines.insert(1, "| " + " | ".join("-" * width for width in widths) + " |")
    return "\n".join(lines)


def speed(runs):
    if not all(runs.get(impl) for impl in IMPLS):
        return None
    upstream, local = (statistics.mean(h for _, h, _ in runs[impl].values()) for impl in IMPLS)
    return upstream, local, (local - upstream) / upstream


def aggregate(seeds):
    """Seeds evaluated at the same steps: steps, mean hours, and mean/min/max return."""
    curves = [curve for curve, _, _ in seeds.values()]
    steps = {tuple(step for step, _, _ in curve) for curve in curves}
    if len(steps) != 1:
        raise ValueError(f"Seeds disagree on evaluation steps: {sorted(steps)}")
    points = list(zip(*curves))
    return (
        list(steps.pop()),
        [statistics.mean(p[1] for p in point) for point in points],
        [statistics.mean(p[2] for p in point) for point in points],
        [min(p[2] for p in point) for point in points],
        [max(p[2] for p in point) for point in points],
    )


def figure(env, runs):
    cfg = ENVS[env]
    plt.rcParams.update(
        {
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 10,
        }
    )
    fig, (ax_time, ax_step) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    for ax in (ax_time, ax_step):
        ax.set_facecolor(SURFACE)
        ax.grid(color="#e4e3df", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel(f"Eval average return ({cfg.episodes} episodes)")
    if env == "humanoid":
        paper = paper_humanoid()
        finals = paper[-1][1]
        ax_time.axhspan(
            min(finals),
            max(finals),
            color=BAND,
            alpha=0.35,
            linewidth=0,
            label=f"Paper @1M, {len(finals)} seeds (min–max)",
        )
        ax_step.fill_between(
            [s / 1e6 for s, _ in paper],
            [min(v) for _, v in paper],
            [max(v) for _, v in paper],
            color=BAND,
            alpha=0.35,
            linewidth=0,
            label=f"Paper, {len(finals)} seeds (min–max)",
        )
        ax_step.plot(
            [s / 1e6 for s, _ in paper],
            [statistics.mean(v) for _, v in paper],
            color=MUTED,
            linewidth=1.2,
            linestyle="--",
            label="Paper, seed mean",
        )
    ends = {}
    for impl, (label, color) in IMPLS.items():
        seeds = runs.get(impl)
        if not seeds:
            continue
        steps, hours, mean, low, high = aggregate(seeds)
        name = f"{label}, {len(seeds)} seed{'s' * (len(seeds) > 1)}"
        style = {
            "color": color,
            "linewidth": 2,
            "marker": "o",
            "markersize": 4,
            "markeredgecolor": SURFACE,
            "markeredgewidth": 1.2,
            "label": name,
        }
        ax_time.plot(hours, mean, **style)
        ax_step.plot([s / 1e6 for s in steps], mean, **style)
        if len(seeds) > 1:
            ax_time.fill_between(hours, low, high, color=color, alpha=0.18, linewidth=0)
            ax_step.fill_between(
                [s / 1e6 for s in steps],
                low,
                high,
                color=color,
                alpha=0.18,
                linewidth=0,
            )
        ends[impl] = statistics.mean(h for _, h, _ in seeds.values())
        ax_time.axvline(ends[impl], color=color, linestyle="--", linewidth=1.2, zorder=1)
    gain = speed(runs)
    if gain:
        upstream, local, ratio = gain
        bottom, top = ax_time.get_ylim()
        ax_time.set_ylim(bottom, top + 0.18 * (top - bottom))
        y = top + 0.08 * (top - bottom)
        ax_time.annotate(
            "",
            xy=(local, y),
            xytext=(upstream, y),
            arrowprops={"arrowstyle": "<->", "color": INK, "linewidth": 1},
        )
        # A wide gap holds the label above the arrow; a narrow one puts it left of both end lines.
        wide = abs(local - upstream) > 0.4 * max(local, upstream)
        ax_time.text(
            (local + upstream) / 2 if wide else min(local, upstream) - 0.01 * max(local, upstream),
            y + 0.015 * (top - bottom) if wide else y,
            f"{cfg.final} steps, seed mean: JAX {local:.2f} h vs PyTorch "
            f"{upstream:.2f} h ({ratio:+.0%})",
            ha="center" if wide else "right",
            va="bottom" if wide else "center",
            color=INK,
            fontsize=8,
        )
    ax_time.set_xlabel("Wall-clock time since launch (hours, seed mean)")
    ax_time.set_title("Eval return vs wall-clock", color=INK, loc="left")
    ax_step.set_xlabel("Environment steps (millions)")
    ax_step.set_title("Eval return vs env steps", color=INK, loc="left")
    for ax in (ax_time, ax_step):
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        f"FlashSAC · {cfg.title} · seed mean, band = min–max · RTX 4080 SUPER (WSL2)",
        color=INK,
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(cfg.figure, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def replace_block(text, name, body):
    pattern = re.compile(
        rf"(<!-- BEGIN generated:{name} -->\n).*?(\n<!-- END generated:{name} -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError(f"{DOC} has no generated block {name!r}")
    # Blank lines inside the markers, as mdformat writes them.
    return pattern.sub(lambda match: f"{match.group(1)}\n{body}\n{match.group(2)}", text)


def main():
    text = DOC.read_text()
    summary = []
    for env, cfg in ENVS.items():
        runs = load(env)
        counts = ", ".join(
            f"{short} {len(runs.get(i, {}))}/{cfg.seeds[i]}"
            for i, short in (("upstream", "PyTorch"), ("jax", "JAX"))
        )
        if env == "humanoid":
            threshold = 0.9 * statistics.mean(paper_humanoid()[-1][1])
            label = "90% of paper"
        else:
            threshold = 0.9 * max(
                statistics.mean(curve[-1][2] for curve, _, _ in seeds.values())
                for seeds in runs.values()
            )
            label = "90% of the best seed-mean final"
        body = table(env, runs, threshold, label)
        gain = speed(runs)
        if gain:
            upstream, local, ratio = gain
            line = (
                f"Seed-mean wall-clock to {cfg.final} steps: JAX {local:.2f} h vs PyTorch "
                f"{upstream:.2f} h ({ratio:+.0%})."
            )
            body += "\n\n" + line
            summary.append(
                f"- {cfg.name}, {cfg.final} steps: {line.split(': ', 1)[1]} "
                f"Seeds finished: {counts}."
            )
        else:
            summary.append(f"- {cfg.name}: runs in progress ({counts}).")
        figure(env, runs)
        text = replace_block(text, env, body)
        print(f"## {cfg.name}\n{body}\n\nfigure: {cfg.figure}\n")
    text = replace_block(text, "summary", "\n".join(summary))
    DOC.write_text(text)
    print("\n".join(summary))


if __name__ == "__main__":
    main()
