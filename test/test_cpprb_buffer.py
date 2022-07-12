from haiku_baselines.common.cpprb_buffers import *

buffer = PrioritizedNstepReplayBuffer(100,[[2]], 1, 1, 3, 0.9,0.4)


for ep in range(10):
	for idx in range(9):
		buffer.add([[ep,idx]],idx,1,[[ep,idx]],False,False)
	buffer.add([[ep,9]],9,1,[[ep,9]],True,True)

samples = buffer.sample(20)
for i in range(20):
	for k in samples:
		if k == 'obses' or k == 'nxtobses':
			s = samples[k][0]
		else:
			s = samples[k]
		print(f'{k} :', s[i],end=' ')
	print()