export CUDA_VISIBLE_DEVICES=1
export DISPLAY=:0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
LR="--learning_rate 0.00003"
TRAIN="--steps 2e6 --worker 32 --batch 32 --mini_batch 256 --gamma 0.995 --lamda 0.95"
MODEL="--node 512 --hidden_n 1"
OPTIONS="--ent_coef 1e-6"
OPTIMIZER="--optimizer rmsprop"

LR="--learning_rate 0.0001"
export CUDA_VISIBLE_DEVICES=1
python run_pg.py --algo A2C $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
LR="--learning_rate 0.00003"
OPTIONS="--ent_coef 1e-3"
#export CUDA_VISIBLE_DEVICES=2
#python run_pg.py --algo PPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
#export CUDA_VISIBLE_DEVICES=3
#python run_pg.py --algo TPPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
