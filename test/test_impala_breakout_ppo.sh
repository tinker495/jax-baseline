export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
RL="--learning_rate 0.00003"
TRAIN="--steps 1e5 --batch 256 --gamma 0.995 --worker 8 --ent_coef 1e-3"
MODEL="--node 512 --hidden_n 1"
OPTIONS="--sample_size 32 --update_freq 10 --val_coef 0.6 --buffer_size 256 --mu_ratio 0.0" # --buffer_size 256
OPTIMIZER="--optimizer rmsprop"

python run_impala.py --algo PPO $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

OPTIONS="--sample_size 32 --update_freq 10 --val_coef 0.6 --buffer_size 256 --mu_ratio 0.2" # --buffer_size 256

python run_impala.py --algo PPO $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

OPTIONS="--sample_size 32 --update_freq 10 --val_coef 0.6 --buffer_size 256 --mu_ratio 0.5" # --buffer_size 256

python run_impala.py --algo PPO $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

OPTIONS="--sample_size 32 --update_freq 10 --val_coef 0.6 --buffer_size 256 --mu_ratio 1.0" # --buffer_size 256

python run_impala.py --algo PPO $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS