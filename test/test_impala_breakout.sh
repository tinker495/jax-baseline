export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
LR="--learning_rate 0.0003"
TRAIN="--steps 1e6 --batch 32 --gamma 0.99 --worker 16 --ent_coef 1e-2"
MODEL="--node 512 --hidden_n 1"
OPTIONS="--sample_size 32 --update_freq 100 --val_coef 0.6 --buffer_size 256" # --buffer_size 256
OPTIMIZER="--optimizer rmsprop"

#export CUDA_VISIBLE_DEVICES=0
#python run_impala.py --algo PPO $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --mu_ratio 0.5

export CUDA_VISIBLE_DEVICES=2
python run_impala.py --algo A2C $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
