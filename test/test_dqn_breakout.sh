export CUDA_VISIBLE_DEVICES=1
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
LR="--learning_rate 0.0001"
TRAIN="--steps 1e6 --batch 32 --train_freq 4 --final_eps 0.01 --learning_starts 20000 --gamma 0.99 --buffer_size 1e6 --exploration_fraction 0.1"
MODEL="--node 512 --hidden_n 1"
OPTIONS="--target_update 1000"
OPTIMIZER="--optimizer adam"

#python run_qnet.py --algo DQN $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python run_qnet.py --algo C51 $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
LR="--learning_rate 0.00005"
#python run_qnet.py --algo QRDQN $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 200
#python run_qnet.py --algo IQN $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python run_qnet.py --algo FQF $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS

LR="--learning_rate 0.0001"
TRAIN="--steps 1e5 --batch 32 --train_freq 1 --final_eps 0.01 --learning_starts 2000 --gamma 0.99 --buffer_size 1e5 --exploration_fraction 0.2"
OPTIONS="--target_update 1000 --gradient_steps 2 --double --dueling --per --n_step 3"
#python run_qnet.py --algo C51 $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python run_qnet.py --algo C51 $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_step 10
python run_qnet.py --algo BBF $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python run_qnet.py --algo SPR $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --soft_reset
#python run_qnet.py --algo SPR $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --hl_gauss
#python run_qnet.py --algo SPR $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --hl_gauss --soft_reset

LR="--learning_rate 0.0001"
TRAIN="--steps 5e5 --batch 32 --train_freq 1 --final_eps 0.01 --learning_starts 2000 --gamma 0.99 --buffer_size 1e5 --exploration_fraction 0.2"
OPTIONS="--target_update 1000 --gradient_steps 2 --double --dueling --per --n_step 3 --noisynet"
#python run_qnet.py --algo C51 $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python run_qnet.py --algo C51 $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_step 10
#python run_qnet.py --algo SPR $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python run_qnet.py --algo SPR $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_step 10 --off_policy_fix
