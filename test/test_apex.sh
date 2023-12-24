export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

#ENV="--env CartPole-v1"
#ENV="--env LunarLander-v2"
#ENV="--env PongNoFrameskip-v4" AsterixNoFrameskip-v4
ENV="--env SpaceInvadersNoFrameskip-v4"
RL="--learning_rate 0.00005"
TRAIN="--steps 2e5 --batch_num 16 --batch_size 512 --target_update 20 --learning_starts 50000 --gamma 0.99 --buffer_size 2e6 --worker 32"
MODEL="--node 512 --hidden_n 0"
OPTIONS="--double --dueling --n_step 3 --initial_eps 0.4 --eps_decay 7"
OPTIMIZER="--optimizer rmsprop"

#python run_apex_qnet.py --algo DQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

OPTIONS="--double --n_step 3 --initial_eps 0.4 --eps_decay 7 --max 10 --min -10"
export CUDA_VISIBLE_DEVICES=1
#python run_apex_qnet.py --algo C51 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

#export CUDA_VISIBLE_DEVICES=2
python run_apex_qnet.py --algo QRDQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
