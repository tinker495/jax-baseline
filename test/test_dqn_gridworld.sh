export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0
ENV="--env ../../env/GridWorld.x86_64"
RL="--learning_rate 0.0002"
TRAIN="--steps 1e5 --batch 32 --train_freq 1 --target_update 1000 --final_eps 0.01 --learning_starts 1000 --gamma 0.995 --buffer_size 1e5 --exploration_fraction 0.3"
MODEL="--node 512 --hidden_n 0"
OPTIONS="--time_scale 1 --capture_frame_rate 30"
OPTIMIZER="--optimizer adamw"
python run_qnet.py --algo DQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS