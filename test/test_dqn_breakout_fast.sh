export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=dummy
ENV="--env LunarLander-v2"
TRAIN="--learning_rate 0.0002 --steps 1e5 --batch 32 --train_freq 1 --target_update 1000 --final_eps 0.01 --learning_starts 1000 --gamma 0.99 --buffer_size 1e5"
MODEL="--node 128 --hidden_n 2"
OPTIONS=""
OPTIMIZER="--optimizer rmsprop"
python run_qnet.py --algo DQN $ENV $TRAIN $MODEL $OPTIMIZER $OPTIONS
python run_qnet.py --algo C51 $ENV $TRAIN $MODEL $OPTIMIZER $OPTIONS --min 0 --max 20
python run_qnet.py --algo QRDQN $ENV $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 200 --delta 0.1
python run_qnet.py --algo IQN $ENV $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 64 --delta 0.1