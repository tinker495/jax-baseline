export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

#ENV="--env CartPole-v1"
#ENV="--env Acrobot-v1"
#ENV="--env LunarLander-v2"
#ENV="--env Pendulum-v1"
#ENV="--env BipedalWalker-v3"
#ENV="--env MinAtar/Breakout-v1"
#ENV="--env MinAtar/SpaceInvaders-v1"
ENV="--env BreakoutNoFrameskip-v4"
RL="--learning_rate 0.00003"
TRAIN="--steps 1e6 --batch 128 --gamma 0.995 --worker 8 --ent_coef 1e-5"
#TRAIN="--steps 1e5 --batch 128 --gamma 0.995 --worker 16 --ent_coef 1e-5"
MODEL="--node 512 --hidden_n 1"
#OPTIONS="--sample_size 1 --update_freq 100" #--buffer_size 32
OPTIONS="--sample_size 1 --update_freq 100 --val_coef 0.6" #--buffer_size 32
OPTIMIZER="--optimizer rmsprop"

python run_impala.py --algo A2C $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python run_impala.py --algo PPO $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS