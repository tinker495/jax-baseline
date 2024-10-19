import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

plt.ioff()
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
div = 20

csvdict = {
    "DQN": "docs/csv/dqn_5m/DQN_signed.csv",
    "C51": "docs/csv/dqn_5m/C51_signed.csv",
    "QRDQN": "docs/csv/dqn_5m/QRDQN_signed.csv",
    "IQN": "docs/csv/dqn_5m/IQN_signed.csv",
    "FQF": "docs/csv/dqn_5m/FQF_signed.csv",
}
dfdict = {}
average_dict = {}
maximum_steps = 0
for key in csvdict:
    df = pd.read_csv(csvdict[key])
    df = df.astype(float)
    max_step = df["Step"].max()
    maximum_steps = max(maximum_steps, max_step)
    split_size = max_step / div
    df["Split_Step"] = df["Step"].apply(lambda x: round(x / split_size))
    averagedf = df.groupby("Split_Step")["Value"].mean()
    averagedf = averagedf.reset_index()
    averagedf.columns = ["Split_Step", "Average_Reward"]
    averagedf["Step"] = averagedf["Split_Step"] * split_size
    average_dict[key] = averagedf
    dfdict[key] = df

for key in dfdict:
    sns.lineplot(data=average_dict[key], x="Step", y="Average_Reward", label=key)
plt.xlabel("Step")
plt.ylabel("Average Return")
plt.xlim(0, maximum_steps)
plt.title("Signed Reward")
plt.legend()

# get all max average rewards
for key in average_dict:
    print(f"{key}, {average_dict[key]['Average_Reward'].max():.2f}")

csvdict = {
    "DQN": "docs/csv/dqn_5m/DQN_original.csv",
    "C51": "docs/csv/dqn_5m/C51_original.csv",
    "QRDQN": "docs/csv/dqn_5m/QRDQN_original.csv",
    "IQN": "docs/csv/dqn_5m/IQN_original.csv",
    "FQF": "docs/csv/dqn_5m/FQF_original.csv",
}
dfdict = {}
average_dict = {}
maximum_steps = 0
for key in csvdict:
    df = pd.read_csv(csvdict[key])
    df = df.astype(float)
    max_step = df["Step"].max()
    maximum_steps = max(maximum_steps, max_step)
    split_size = max_step / div
    df["Split_Step"] = df["Step"].apply(lambda x: round(x / split_size))
    averagedf = df.groupby("Split_Step")["Value"].mean()
    averagedf = averagedf.reset_index()
    averagedf.columns = ["Split_Step", "Average_Reward"]
    averagedf["Step"] = averagedf["Split_Step"] * split_size
    average_dict[key] = averagedf
    dfdict[key] = df

plt.subplot(1, 2, 2)

for key in dfdict:
    sns.lineplot(data=average_dict[key], x="Step", y="Average_Reward", label=key)
plt.xlabel("Step")
plt.ylabel("Average Return")
plt.xlim(0, maximum_steps)
plt.title("Original Reward")
plt.legend()


plt.savefig("docs/figures/dqn_breakout_5m.png")

# get all max average rewards
for key in average_dict:
    print(f"{key}, {average_dict[key]['Average_Reward'].max():.2f}")
