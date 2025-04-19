export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

EXPERIMENT_NAME="--experiment_name DQN_Acrobot-v1"
ENV="--env Acrobot-v1"
LR="--learning_rate 0.0001"
TRAIN="--steps 5e5 --batch 32 --train_freq 4 --final_eps 0.01 --learning_starts 20000 --gamma 0.99 --buffer_size 1e5 --exploration_fraction 0.2"
MODEL="--node 512 --hidden_n 1"
OPTIONS="--target_update 100"
OPTIMIZER="--optimizer adopt"

python run_qnet.py --algo DQN $EXPERIMENT_NAME $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
python run_qnet.py --algo IQN $EXPERIMENT_NAME $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
python run_qnet.py --algo FQF $EXPERIMENT_NAME $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
