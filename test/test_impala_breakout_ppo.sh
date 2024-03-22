export CUDA_VISIBLE_DEVICES=1
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
LR="--learning_rate 0.00003"
TRAIN="--steps 1e5 --batch 32 --gamma 0.99 --worker 16 --ent_coef 1e-4"
MODEL="--node 512 --hidden_n 1"
OPTIONS="--sample_size 32 --update_freq 100 --val_coef 0.6 --buffer_size 256" # --buffer_size 256 " # --buffer_size 256
OPTIMIZER="--optimizer rmsprop"

python run_impala.py --algo PPO $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
