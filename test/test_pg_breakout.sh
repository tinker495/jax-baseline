export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

EXPERIMENT_NAME="--experiment_name PG_Breakout"
ENV="--env BreakoutNoFrameskip-v4 --worker 32"
LR="--learning_rate 0.00003"
TRAIN="--steps 1e7 --batch 128 --mini_batch 256 --gamma 0.995 --lamda 0.95"
MODEL="--node 512 --hidden_n 1"
OPTIONS="--ent_coef 1e-6"
OPTIMIZER="--optimizer rmsprop"

LR="--learning_rate 0.001"
#python run_pg.py --algo A2C $EXPERIMENT_NAME $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
LR="--learning_rate 0.0003"
OPTIONS="--ent_coef 1e-2 --epoch_num 16"
python run_pg.py --algo SPO $EXPERIMENT_NAME $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
python run_pg.py --algo TPPO $EXPERIMENT_NAME $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
python run_pg.py --algo PPO $EXPERIMENT_NAME $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
