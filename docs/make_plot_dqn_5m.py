from plot_utils import plot_panel, plt

plt.figure(figsize=(15, 7))

algorithms = {
    "DQN": "DQN",
    "C51": "C51",
    "QRDQN": "QRDQN",
    "IQN": "IQN",
    "FQF": "FQF",
}
for position, reward_type in enumerate(("signed", "original"), start=1):
    plot_panel(
        position,
        {
            label: f"docs/csv/dqn_5m/{filename}_{reward_type}.csv"
            for label, filename in algorithms.items()
        },
        divisions=20,
        ylabel="Average Return",
        title=f"{reward_type.title()} Reward",
    )

plt.savefig("docs/figures/dqn_breakout_5m.png")
