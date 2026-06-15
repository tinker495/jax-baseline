import os
from pathlib import Path

import numpy as np


def save_obs(obs, path):
    import cv2
    from tqdm import tqdm

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        # remove all files in path
        files = os.listdir(path)
        for f in files:
            os.remove(os.path.join(path, f))
    for i in tqdm(range(len(obs))):
        img = np.asarray(obs[i])
        print(img.shape)
        for j in range(4):
            cv2.imwrite(os.path.join(path, f"{i}_{j}.png"), img[:, :, j])


def main():
    from env_builder.atari_wrappers import get_env_type, make_wrap_atari

    env_name = "BreakoutNoFrameskip-v4"
    env_type, env_id = get_env_type(env_name)
    if env_type != "atari_env":
        raise RuntimeError(f"{env_name} resolved to unsupported env type: {env_type}")

    env = make_wrap_atari(env_name, clip_rewards=True)
    obs = []
    for i in range(1):
        obs_ep = [env.reset()[0]]
        for j in range(1):
            obs_ep.append(env.step(env.action_space.sample())[0])
        obs.extend(obs_ep)

    save_obs(obs, str(Path(__file__).resolve().parent / "test"))


if __name__ == "__main__":
    main()
