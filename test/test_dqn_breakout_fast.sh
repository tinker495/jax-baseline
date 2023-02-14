export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=dummy
ENV="--env BreakoutNoFrameskip-v4 --clip_rewards"
TRAIN="--learning_rate 0.0003 --steps 3e6 --batch 16 --train_freq 1 --target_update 500 --final_eps 0.05 --learning_starts 1000 --gamma 0.995 --buffer_size 1e6 --exploration_fraction 0.3"
OPTIMIZER="--optimizer rmsprop"
python run_qnet.py --algo DQN $ENV $TRAIN $OPTIMIZER
python run_qnet.py --algo C51 $ENV $TRAIN $OPTIMIZER --min 0 --max 30
python run_qnet.py --algo QRDQN $ENV $TRAIN $OPTIMIZER --n_support 200 --delta 0.1
python run_qnet.py --algo IQN $ENV $TRAIN $OPTIMIZER --n_support 64 --delta 0.1
OPTIMIZER="adam"
python run_qnet.py --algo DQN $ENV $TRAIN $OPTIMIZER
python run_qnet.py --algo C51 $ENV $TRAIN $OPTIMIZER --min 0 --max 30
python run_qnet.py --algo QRDQN $ENV $TRAIN $OPTIMIZER --n_support 200 --delta 0.1
python run_qnet.py --algo IQN $ENV $TRAIN $OPTIMIZER --n_support 64 --delta 0.1
OPTIMIZER="adamw"
python run_qnet.py --algo DQN $ENV $TRAIN $OPTIMIZER
python run_qnet.py --algo C51 $ENV $TRAIN $OPTIMIZER --min 0 --max 30
python run_qnet.py --algo QRDQN $ENV $TRAIN $OPTIMIZER --n_support 200 --delta 0.1
python run_qnet.py --algo IQN $ENV $TRAIN $OPTIMIZER --n_support 64 --delta 0.1