export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
LR="--learning_rate 0.0001"
OPTIMIZER="--optimizer adam"
MODEL="--node 512 --hidden_n 1"

TRAIN="--steps 1e5 --batch 32 --train_freq 4 --final_eps 0.01 --learning_starts 2000 --gamma 0.99 --buffer_size 1e5 --exploration_fraction 0.2"
OPTIONS="--target_update 1000 --gradient_steps 1 --double --dueling --per --n_step 10 --noisynet"
python run_qnet.py --algo DQN $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
python run_qnet.py --algo C51 $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS

# High replay ratio
TRAIN="--steps 1e5 --batch 32 --train_freq 1 --final_eps 0.01 --learning_starts 2000 --gamma 0.99 --buffer_size 1e5 --exploration_fraction 0.2"
OPTIONS="--target_update 1000 --gradient_steps 2 --double --dueling --per --n_step 10 --noisynet"
python run_qnet.py --algo DQN $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
python run_qnet.py --algo C51 $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
OPTIONS="--gradient_steps 2"
python run_qnet.py --algo SPR $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
python run_qnet.py --algo SPR $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --soft_reset
python run_qnet.py --algo BBF $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS