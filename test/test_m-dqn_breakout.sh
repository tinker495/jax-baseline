export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
LR="--learning_rate 0.0001"
TRAIN="--steps 5e6 --batch 32 --train_freq 4 --final_eps 0.01 --learning_starts 20000 --gamma 0.99 --buffer_size 1e5 --exploration_fraction 0.2"
MODEL="--node 512 --hidden_n 1"
OPTIONS="--target_update 1000"
OPTIMIZER="--optimizer adam"

python run_qnet.py --algo DQN $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
python run_qnet.py --algo DQN $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --munchausen
python run_qnet.py --algo C51 $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
python run_qnet.py --algo C51 $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --munchausen
python run_qnet.py --algo QRDQN $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 200
python run_qnet.py --algo QRDQN $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 200 --munchausen
python run_qnet.py --algo IQN $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
python run_qnet.py --algo IQN $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --munchausen
python run_qnet.py --algo FQF $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
python run_qnet.py --algo FQF $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --munchausen
