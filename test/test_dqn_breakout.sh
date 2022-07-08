export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=dummy
python run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --steps 1e7 --batch 64 --target_update 2000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 1000 --gamma 0.995 --buffer_size 100000 --exploration_fraction 0.3
python run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --steps 1e7 --batch 64 --target_update 2000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 1000 --gamma 0.995 --buffer_size 100000 --exploration_fraction 0.3 --n_step 5 --per
python run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --steps 1e7 --batch 64 --target_update 2000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 1000 --gamma 0.995 --buffer_size 100000 --exploration_fraction 0.3 --n_step 5 --per --double --dueling
