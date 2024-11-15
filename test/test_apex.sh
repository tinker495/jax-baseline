export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
LR="--learning_rate 0.00005"
TRAIN="--steps 5e4 --batch_num 16 --batch_size 512 --target_update 1 --learning_starts 50000 --gamma 0.99 --buffer_size 1e5 --worker 8"
MODEL="--node 512 --hidden_n 1"
OPTIONS="--double --dueling --n_step 3 --initial_eps 0.4 --eps_decay 7"
OPTIMIZER="--optimizer adam"


#OPTIONS="--double --dueling --n_step 3 --initial_eps 0.4 --eps_decay 7 --max 10 --min -10"
#python run_apex_qnet.py --algo C51 $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS

OPTIONS="--double --dueling --n_step 3 --initial_eps 0.4 --eps_decay 7 --n_support 64"
python run_apex_qnet.py --algo QRDQN $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS

OPTIONS="--double --n_step 3 --initial_eps 0.4 --eps_decay 7"
python run_apex_qnet.py --algo DQN $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
