export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
RL="--learning_rate 0.0001"
TRAIN="--steps 1e7 --batch 32 --train_freq 4 --target_update 1000 --final_eps 0.01 --learning_starts 20000 --gamma 0.99 --buffer_size 1e6 --exploration_fraction 0.1"
MODEL="--node 512 --hidden_n 0"
OPTIONS=""
OPTIMIZER="--optimizer adam"

python run_qnet.py --algo DQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS &
export CUDA_VISIBLE_DEVICES=0
python run_qnet.py --algo C51 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS &
export CUDA_VISIBLE_DEVICES=1
python run_qnet.py --algo QRDQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 200 &
export CUDA_VISIBLE_DEVICES=2
python run_qnet.py --algo IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

#MODEL="--node 512 --hidden_n 1 --double --noisynet --per --n_step 2"
#python run_qnet.py --algo C51 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS  --min -10 --max 10 &
#python run_qnet.py --algo QRDQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 200 &
#export CUDA_VISIBLE_DEVICES=1
#python run_qnet.py --algo IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 64 &
#export CUDA_VISIBLE_DEVICES=2
#python run_qnet.py --algo IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --n_support 64 --CVaR 0.2
