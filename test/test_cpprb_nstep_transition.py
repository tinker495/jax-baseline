from haiku_baselines.common.cpprb_buffers import ReplayBuffer, MultiPrioritizedReplayBuffer
import numpy as np

n=2
gamma = 0.995
mprb = MultiPrioritizedReplayBuffer(200, [[4]], 0.6, 1, n_step=n, gamma=gamma)
gloabal_buffer, env_dict, n_s = mprb.buffer_info()
rb = ReplayBuffer(100, env_dict = env_dict, n_s=n_s)
true_done = True

for i in range(11):
    if i % 20 == 10:
        rb.add([np.arange(i,i+4)], 1, 1, [np.arange(i+4,i+8)], true_done, False)
        rb.episode_end()
    else:
        rb.add([np.arange(i,i+4)], 1, 1, [np.arange(i+4,i+8)], False, False)
#

transition = rb.get_buffer()
rb.clear()
#print(transition)

gloabal_buffer.add(**transition)

mptransition = gloabal_buffer.get_all_transitions()
#print(mptransition)

for i in range(len(mptransition['done'])):
    st = ""
    for k in mptransition.keys():
        st += f"{k}: {mptransition[k][i]}, "
    print(st)

for i in range(n):
    d = mptransition[k][-(n - i)]
    print((1 - d) * np.power(gamma,(n-i)))

print((1 - true_done) * np.power(gamma,n))

#print()
