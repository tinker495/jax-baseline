pip install .. -q
export SDL_VIDEODRIVER=dummy
export CUDA_VISIBLE_DEVICES=0
python3 run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0002 --steps 1e7 --batch 32 --train_freq 4 --target_update 5000 --node 256 --hidden_n 1 --final_eps 0.05 --exploration_fraction 0.8 --learning_starts 10000 --gamma 0.995 --buffer_size 1e6 --clip_rewards

python3 run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0002 --steps 1e7 --batch 32 --train_freq 4 --target_update 5000 --node 256 --hidden_n 1 --final_eps 0.05 --exploration_fraction 0.8 --learning_starts 10000 --gamma 0.995 --buffer_size 1e6 --clip_rewards --per

python3 run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0002 --steps 1e7 --batch 32 --train_freq 4 --target_update 5000 --node 256 --hidden_n 1 --final_eps 0.05 --exploration_fraction 0.8 --learning_starts 10000 --gamma 0.995 --buffer_size 1e6 --clip_rewards --n_step 4
