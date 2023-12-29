export CUDA_VISIBLE_DEVICES=1
export DISPLAY=:0

pip install -q ..

ENV="--env ../../env/GridWorld.x86_64"
LR="--learning_rate 0.0002"
TRAIN="--steps 5e4 --batch 32 --train_freq 1 --target_update 1000 --final_eps 0.01 --learning_starts 1000 --gamma 0.9 --buffer_size 1e5 --exploration_fraction 0.2"
MODEL="--node 512 --hidden_n 2"
OPTIONS="--time_scale 1 --capture_frame_rate 30"
OPTIMIZER="--optimizer adamw"
python run_qnet.py --algo DQN $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
