from plot_utils import plot_panel, plt

plt.figure(figsize=(15, 7))

panels = (
    {
        "TD3": "TD3",
        "SAC": "SAC",
        "TQC": "TQC",
        "TD7": "TD7",
        "CrossQ": "CrossQ",
    },
    {
        "Simba + TD3": "Simba_TD3",
        "Simba + SAC": "Simba_SAC",
        "Simba + TQC": "Simba_TQC",
        "Simba + TD7": "Simba_TD7",
        "RSNorm + CrossQ": "Simba_CrossQ",
    },
)
for position, algorithms in enumerate(panels, start=1):
    plot_panel(
        position,
        {
            label: f"docs/csv/dpg_humanoid_5m/{filename}.csv"
            for label, filename in algorithms.items()
        },
        divisions=50,
        ylabel="Average Reward",
        title="Average Reward",
        ylim=(0, 15000),
    )

plt.savefig("docs/figures/dpg_Humanoid-v4-5m.png")
