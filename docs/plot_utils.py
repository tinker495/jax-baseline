import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
plt.ioff()


def plot_panel(position, csv_paths, *, divisions, ylabel, title, ylim=None):
    plt.subplot(1, 2, position)
    averages = {}
    maximum_steps = 0

    for label, path in csv_paths.items():
        frame = pd.read_csv(path).astype(float)
        max_step = frame["Step"].max()
        maximum_steps = max(maximum_steps, max_step)
        split_size = max_step / divisions
        frame["Split_Step"] = frame["Step"].apply(lambda step: round(step / split_size))
        average = frame.groupby("Split_Step")["Value"].mean().reset_index()
        average.columns = ["Split_Step", "Average_Reward"]
        average["Step"] = average["Split_Step"] * split_size
        averages[label] = average

    for label, average in averages.items():
        sns.lineplot(data=average, x="Step", y="Average_Reward", label=label)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.xlim(0, maximum_steps)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    plt.legend()

    for label, average in averages.items():
        print(f"{label}, {average['Average_Reward'].max():.2f}")
