export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
RL="--learning_rate 0.0003"
TRAIN="--steps 1e5 --batch 32 --gamma 0.995 --worker 8 --ent_coef 1e-4"
MODEL="--node 512 --hidden_n 1"
OPTIONS="--sample_size 256 --update_freq 2 --val_coef 0.6 --buffer_size 1024 --mu_ratio 0.5" # --buffer_size 256
OPTIMIZER="--optimizer adam"

python run_impala.py --algo PPO $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS