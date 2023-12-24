export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
RL="--learning_rate 0.00003"
TRAIN="--steps 2e6 --worker 32 --batch 32 --mini_batch 256 --gamma 0.995 --lamda 0.95 --ent_coef 1e-3"
MODEL="--node 512 --hidden_n 1"
OPTIONS=""
OPTIMIZER="--optimizer rmsprop"

python run_pg.py --algo PPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
