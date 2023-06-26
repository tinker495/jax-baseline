export CUDA_VISIBLE_DEVICES=1
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
RL="--learning_rate 0.00003"
TRAIN="--steps 1e6 --worker 32 --batch 128 --mini_batch 256 --gamma 0.995 --lamda 0.95 --ent_coef 1e-6 --gae_normalize"
MODEL="--node 512 --hidden_n 1"
OPTIONS=""
OPTIMIZER="--optimizer adam"
python run_pg.py --algo PPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER