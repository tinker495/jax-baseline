export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
RL="--learning_rate 0.0001"
TRAIN="--steps 5e5 --batch 32 --train_freq 1 --target_update 250 --final_eps 0.01 --learning_starts 10000 --gamma 0.995 --buffer_size 1e5 --exploration_fraction 0.1"
MODEL="--node 512 --hidden_n 1"
OPTIONS="--double --dueling"
OPTIMIZER="--optimizer adam"

python run_qnet.py --algo DQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --min 0 --max 30

OPTIONS="--double --dueling --n_step 3"

python run_qnet.py --algo DQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --min 0 --max 30

#python run_qnet.py --algo C51 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --min 0 --max 30
#python run_qnet.py --algo QRDQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 200 --delta 0.1
#python run_qnet.py --algo IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 64 --delta 0.1
