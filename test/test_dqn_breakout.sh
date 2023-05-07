export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
RL="--learning_rate 0.0002"
TRAIN="--steps 2e6 --batch 32 --train_freq 1 --target_update 1000 --final_eps 0.001 --learning_starts 20000 --gamma 0.99 --buffer_size 1e5 --exploration_fraction 0.025"
MODEL="--node 512 --hidden_n 1"
OPTIONS=""
OPTIMIZER="--optimizer adam"

#python run_qnet.py --algo DQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python run_qnet.py --algo C51 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --min -5 --max 10
#python run_qnet.py --algo QRDQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 200 --delta 1
#python run_qnet.py --algo IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 64 --delta 1

OPTIONS="--double --dueling --n_step 3 --per --munchausen"

#python run_qnet.py --algo DQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python run_qnet.py --algo C51 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --min -5 --max 10
#python run_qnet.py --algo QRDQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 200 --delta 1
python run_qnet.py --algo IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 64 --delta 1

#python run_qnet.py --algo DQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python run_qnet.py --algo C51 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --min 0 --max 30
#python run_qnet.py --algo QRDQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 200 --delta 0.1
#python run_qnet.py --algo IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 64 --delta 0.1
