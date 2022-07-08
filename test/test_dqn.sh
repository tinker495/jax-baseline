export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=dummy
#python run_qnet.py --algo DQN --env LunarLander-v2 --steps 1e6 --batch 32 --target_update 5000 --node 512 --hidden_n 2 --final_eps 0.1 --learning_starts 50000 --gamma 0.99 --buffer_size 1e6 --exploration_fraction 0.3 --train_freq 1 --learning_rate 1e-4 &
#python run_qnet.py --algo QRDQN --env LunarLander-v2 --steps 1e6 --batch 32 --target_update 5000 --node 512 --hidden_n 2 --final_eps 0.1 --learning_starts 50000 --gamma 0.99 --buffer_size 1e6 --exploration_fraction 0.3 --train_freq 1 --n_support 64 --learning_rate 1e-4
#python run_qnet.py --algo IQN --env BreakoutNoFrameskip-v4 --steps 1e7 --batch 32 --target_update 10000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 50000 --gamma 0.99 --buffer_size 400000 --exploration_fraction 0.2 --CVaR 1.0  --n_support 64 &
#python run_qnet.py --algo IQN --env BreakoutNoFrameskip-v4 --steps 1e7 --batch 32 --target_update 10000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 50000 --gamma 0.99 --buffer_size 400000 --exploration_fraction 0.2 --CVaR 0.3  --n_support 64
#python run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --steps 1e7 --batch 64 --target_update 2000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 1000 --gamma 0.995 --buffer_size 100000 --exploration_fraction 0.3
python run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --steps 1e7 --batch 64 --target_update 2000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 1000 --gamma 0.995 --buffer_size 100000 --exploration_fraction 0.3 --n_step 5 --per
#python run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --steps 1e7 --batch 32 --target_update 3000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 50000 --gamma 0.99 --buffer_size 400000 --exploration_fraction 0.2