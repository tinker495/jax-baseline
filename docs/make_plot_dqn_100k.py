import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

plt.ioff()

plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
div = 50

csvdict = {
    "Rainbow(DQN)": "docs/csv/dqn_100k/DQN_100k_signed.csv",
    "Rainbow": "docs/csv/dqn_100k/C51_100k_signed.csv",
    "DER(DQN)": "docs/csv/dqn_100k/DQN_rr2_100k_signed.csv",
    "DER": "docs/csv/dqn_100k/C51_rr2_100k_signed.csv",
    "SPR": "docs/csv/dqn_100k/SPR_100k_signed.csv",
    "SR-SPR": "docs/csv/dqn_100k/SR-SPR_100k_signed.csv",
    "BBF": "docs/csv/dqn_100k/BBF_100k_signed.csv",
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
    "Rainbow(DQN)": "docs/csv/dqn_100k/DQN_100k_original.csv",
    "Rainbow": "docs/csv/dqn_100k/C51_100k_original.csv",
    "DER(DQN)": "docs/csv/dqn_100k/DQN_rr2_100k_original.csv",
    "DER": "docs/csv/dqn_100k/C51_rr2_100k_original.csv",
    "SPR": "docs/csv/dqn_100k/SPR_100k_original.csv",
    "SR-SPR": "docs/csv/dqn_100k/SR-SPR_100k_original.csv",
    "BBF": "docs/csv/dqn_100k/BBF_100k_original.csv",
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


plt.savefig("docs/figures/dqn_breakout_100k.png")

# get all max average rewards
for key in average_dict:
    print(f"{key}, {average_dict[key]['Average_Reward'].max():.2f}")
