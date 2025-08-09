export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

pip install -q ..

EXPERIMENT="--experiment_name PG_LunarLander"
ENV="--env LunarLander-v3 --worker 32"
LR="--learning_rate 0.00003"
TRAIN="--steps 1e6 --batch 128 --mini_batch 256 --gamma 0.995 --lamda 0.95"
MODEL="--node 128 --hidden_n 2"
OPTIONS="--ent_coef 1e-6"
OPTIMIZER="--optimizer rmsprop"

LR="--learning_rate 0.001"
python run_pg.py --algo A2C $EXPERIMENT $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
LR="--learning_rate 0.0003"
OPTIONS="--ent_coef 1e-3"
python run_pg.py --algo PPO $EXPERIMENT $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
python run_pg.py --algo TPPO $EXPERIMENT $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
