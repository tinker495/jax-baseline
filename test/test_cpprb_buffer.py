from haiku_baselines.common.cpprb_buffers import *

buffer = PrioritizedNstepReplayBuffer(100,[[2]], 1, 1, 3, 0.9,0.4)


for ep in range(1):
	for idx in range(9):
		buffer.add([[ep,idx]],idx,1,[[ep,idx]],False,False)
	buffer.add([[ep,9]],9,1,[[ep,9]],True,True)

samples = buffer.sample(10)
for i in range(10):
	for k in samples:
		if k == 'obses' or k == 'nxtobses':
			s = samples[k][0]
		else:
			s = samples[k]
		print(f'{k} :', s[i],end=' ')
	print()
    
samples = buffer.sample(10)
for i in range(10):
	for k in samples:
		if k == 'obses' or k == 'nxtobses':
			s = samples[k][0]
		else:
			s = samples[k]
		print(f'{k} :', s[i],end=' ')
	print()
    
import numpy as np

buffer.update_priorities(samples['indexes'],samples['indexes'])

samples = buffer.sample(10)
for i in range(10):
	for k in samples:
		if k == 'obses' or k == 'nxtobses':
			s = samples[k][0]
		else:
			s = samples[k]
		print(f'{k} :', s[i],end=' ')
	print()
                         
                         