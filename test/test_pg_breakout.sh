export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=dummy
ENV="--env BreakoutNoFrameskip-v4"
RL="--learning_rate 0.0002"
TRAIN="--steps 2e6 --steps 2e6 --worker 8 --batch 128 --mini_batch 256 --gamma 0.995 --lamda 0.95 --ent_coef 0.01 --gae_normalize"
MODEL="--node 512 --hidden_n 0"
OPTIONS=""
OPTIMIZER="--optimizer adamw"
python run_pg.py --algo PPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
TRAIN="--steps 2e6 --worker 8 --batch 32 --node 512 --hidden_n 1 --ent_coef 5e-2"
python run_pg.py --algo A2C $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER