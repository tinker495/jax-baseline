export CUDA_VISIBLE_DEVICES=1
export DISPLAY=:0

pip install -q ..

ENV="--env ../../env/GridWorld.x86_64"
RL="--learning_rate 0.0002"
TRAIN="--steps 5e4 --batch 128 --mini_batch 32 --gamma 0.9 --lamda 0.95 --ent_coef 0.0001 --gae_normalize"
MODEL="--node 512 --hidden_n 2"
OPTIONS="--time_scale 1 --capture_frame_rate 30"
OPTIMIZER="--optimizer adamw"
python run_pg.py --algo PPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER