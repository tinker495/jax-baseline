export CUDA_VISIBLE_DEVICES=1
export DISPLAY=:0

pip install -q ..

#ENV="--env CartPole-v1"
#ENV="--env LunarLander-v2"
#ENV="--env PongNoFrameskip-v4"
ENV="--env BreakoutNoFrameskip-v4"
RL="--learning_rate 0.0001"
TRAIN="--steps 1e6 --batch 256 --target_update 250 --learning_starts 10000 --gamma 0.995 --buffer_size 1e5 --worker 8"
MODEL="--node 512 --hidden_n 1"
OPTIONS="--double --dueling --n_step 3 --initial_eps 0.1 --eps_decay 3"
OPTIMIZER="--optimizer adam"

python run_apex_qnet.py --algo DQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS