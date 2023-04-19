export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0
ENV="--env ../../env/GridWorld.x86_64"
RL="--learning_rate 0.0002"
TRAIN="--steps 1e5 --batch 128 --mini_batch 32 --gamma 0.995 --lamda 0.95 --ent_coef 0.01 --gae_normalize"
MODEL="--node 512 --hidden_n 0"
OPTIONS="--time_scale 1 --capture_frame_rate 30"
OPTIMIZER="--optimizer adamw"
python run_pg.py --algo PPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER