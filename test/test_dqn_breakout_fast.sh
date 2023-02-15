export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=dummy
ENV="--env LunarLander-v2"
TRAIN="--learning_rate 0.0002 --steps 5e5 --batch 32 --train_freq 1 --target_update 1000 --final_eps 0.1 --learning_starts 1000 --gamma 0.99 --buffer_size 1e5 --exploration_fraction 0.6"
MODEL="--node 512 --hidden_n 2"
OPTIONS=""
OPTIMIZER="--optimizer rmsprop"
python run_qnet.py --algo DQN $ENV $TRAIN $MODEL $OPTIMIZER $OPTIONS
python run_qnet.py --algo C51 $ENV $TRAIN $MODEL $OPTIMIZER $OPTIONS --min -30 --max 30
TRAIN="--learning_rate 0.00005 --steps 5e5 --batch 32 --train_freq 1 --target_update 1000 --final_eps 0.1 --learning_starts 1000 --gamma 0.99 --buffer_size 1e5 --exploration_fraction 0.6"
python run_qnet.py --algo QRDQN $ENV $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 200 --delta 0.1
python run_qnet.py --algo IQN $ENV $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 64 --delta 0.1