from plot_utils import plot_panel, plt

plt.figure(figsize=(15, 7))

algorithms = {
    "Rainbow(DQN)": "DQN_100k",
    "Rainbow": "C51_100k",
    "DER(DQN)": "DQN_rr2_100k",
    "DER": "C51_rr2_100k",
    "SPR": "SPR_100k",
    "SR-SPR": "SR-SPR_100k",
    "BBF": "BBF_100k",
}
for position, reward_type in enumerate(("signed", "original"), start=1):
    plot_panel(
        position,
        {
            label: f"docs/csv/dqn_100k/{filename}_{reward_type}.csv"
            for label, filename in algorithms.items()
        },
        divisions=50,
        ylabel="Average Return",
        title=f"{reward_type.title()} Reward",
    )

plt.savefig("docs/figures/dqn_breakout_100k.png")
