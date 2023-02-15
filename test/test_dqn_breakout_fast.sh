export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=dummy
ENV="--env BreakoutNoFrameskip-v4 --clip_rewards"
TRAIN="--learning_rate 0.00025 --steps 3e6 --batch 64 --train_freq 1 --target_update 1000 --final_eps 0.05 --learning_starts 1000 --gamma 0.99 --buffer_size 1e6 --exploration_fraction 0.3"
MODEL="--node 512 --hidden_n 3"
OPTIONS=""
OPTIMIZER="--optimizer rmsprop"
python run_qnet.py --algo DQN $ENV $TRAIN $MODEL $OPTIMIZER $OPTIONS
python run_qnet.py --algo C51 $ENV $TRAIN $MODEL $OPTIMIZER $OPTIONS --min 0 --max 20
python run_qnet.py --algo QRDQN $ENV $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 200 --delta 0.1
python run_qnet.py --algo IQN $ENV $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 64 --delta 0.1