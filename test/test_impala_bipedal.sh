export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env BipedalWalker-v3"
RL="--learning_rate 0.00003"
TRAIN="--steps 5e5 --batch 32 --gamma 0.995 --worker 8 --ent_coef 1e-3"
MODEL="--node 512 --hidden_n 2"
OPTIONS="--sample_size 8 --update_freq 100 --val_coef 0.6 --buffer_size 32" # --buffer_size 256
OPTIMIZER="--optimizer rmsprop"

#python run_impala.py --algo A2C $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

OPTIMIZER="--optimizer rmsprop"
python run_impala.py --algo PPO $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

OPTIMIZER="--optimizer adam"
python run_impala.py --algo PPO $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
