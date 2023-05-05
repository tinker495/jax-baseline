import gym


env_name = "BreakoutNoFrameskip-v4"

from haiku_baselines.common.atari_wrappers import make_wrap_atari,get_env_type
env_type, env_id = get_env_type(env_name)
if env_type == 'atari_env':
    env = make_wrap_atari(env_name,clip_rewards=True)



#save observation video in test folder

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def save_obs(obs, path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        #remove all files in path
        files = os.listdir(path)
        for f in files:
            os.remove(os.path.join(path, f))
    for i in tqdm(range(len(obs))):
        img = np.asarray(obs[i])
        #img = np.mean(img, axis=-1, keepdims=True)
        # resize 512 512
        #img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        print(img.shape)
        for j in range(4):
            cv2.imwrite(os.path.join(path, f'{i}_{j}.png'), img[:,:,j])
        #cv2.imwrite(os.path.join(path, f'{i}.png'), obs[i])



obs = []
for i in range(1):
    obs_ep = [env.reset()[0]]
    for j in range(1):
        obs_ep.append(env.step(env.action_space.sample())[0])
    obs.extend(obs_ep)

save_obs(obs, 'test/test')