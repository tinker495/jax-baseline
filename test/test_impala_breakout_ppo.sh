export CUDA_VISIBLE_DEVICES=4
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
RL="--learning_rate 0.00003"
TRAIN="--steps 5e6 --batch 32 --gamma 0.995 --worker 16 --ent_coef 1e-3"
MODEL="--node 512 --hidden_n 1"
OPTIONS="--sample_size 8 --update_freq 100 --val_coef 0.6 --buffer_size 2048 --mu_ratio 0.05" # --buffer_size 256 " # --buffer_size 256
OPTIMIZER="--optimizer rmsprop"

python run_impala.py --algo PPO $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS