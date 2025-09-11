export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

EXPERIMENT_NAME="--experiment_name DQN_Breakout100k"
ENV="--env BreakoutNoFrameskip-v4"
LR="--learning_rate 0.0001"
OPTIMIZER="--optimizer adamw"
MODEL="--node 512 --hidden_n 1"

TRAIN="--steps 1e5 --batch 32 --train_freq 4 --final_eps 0.001 --learning_starts 1600 --gamma 0.99 --buffer_size 1e5 --exploration_fraction 0.4"
OPTIONS="--target_update 1000 --gradient_steps 1 --double --dueling --per --n_step 3 --noisynet"
python run_qnet.py --algo DQN $EXPERIMENT_NAME $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
python run_qnet.py --algo C51 $EXPERIMENT_NAME $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS

# High replay ratio
TRAIN="--steps 1e5 --batch 32 --train_freq 1 --final_eps 0.001 --learning_starts 1600 --gamma 0.99 --buffer_size 1e5 --exploration_fraction 0.4"
OPTIONS="--target_update 1000 --gradient_steps 2 --double --dueling --per --n_step 3 --noisynet"
python run_qnet.py --algo DQN $EXPERIMENT_NAME $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
python run_qnet.py --algo C51 $EXPERIMENT_NAME $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
OPTIONS="--gradient_steps 2"
python run_qnet.py --algo SPR $EXPERIMENT_NAME $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
python run_qnet.py --algo SPR $EXPERIMENT_NAME $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --scaled_by_reset
MODEL="--node 2048 --hidden_n 1"
python run_qnet.py --algo BBF $EXPERIMENT_NAME $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS

# Super high replay ratio
OPTIMIZER="--optimizer adamw"
OPTIONS="--gradient_steps 8"
python run_qnet.py --algo BBF $EXPERIMENT_NAME $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
