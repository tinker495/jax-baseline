pip install .. -q
export SDL_VIDEODRIVER=dummy
export CUDA_VISIBLE_DEVICES=0
#python run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0002 --steps 1e7 --batch 32 --train_freq 1 --target_update 10000 --node 512 --hidden_n 1 --final_eps 0.05 --learning_starts 20000 --gamma 0.99 --buffer_size 2e5 --munchausen&
export CUDA_VISIBLE_DEVICES=1
python run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0002 --steps 1e7 --batch 32 --train_freq 4 --target_update 10000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 20000 --gamma 0.995 --buffer_size 2e5
#python run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0001 --steps 1e7 --batch 32 --train_freq 4 --target_update 10000 --node 512 --hidden_n 1 --final_eps 0.05 --learning_starts 20000 --gamma 0.99 --buffer_size 5e5 --exploration_fraction 0.3 --n_step 3
#python run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0001 --steps 1e7 --batch 32 --train_freq 4 --target_update 10000 --node 512 --hidden_n 1 --final_eps 0.05 --learning_starts 20000 --gamma 0.99 --buffer_size 5e5 --exploration_fraction 0.3 --per --n_step 3
#zip log.zip -r log/

#python run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0002 --steps 1e7 --batch 32 --train_freq 4 --target_update 10000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 20000 --gamma 0.995 --buffer_size 2e5 --clip_rewards --double --dueling --per --munchausen

#python run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0002 --steps 1e7 --batch 32 --train_freq 4 --target_update 10000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 20000 --gamma 0.995 --buffer_size 2e5 --double --dueling --per --n_step 3 --noisynet