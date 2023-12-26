export CUDA_VISIBLE_DEVICES=1
export DISPLAY=:0

pip install -q ..

#ENV="--env BipedalWalker-v3"
ENV="--env Walker2d-v4"
#"Walker2d-v4"
RL="--learning_rate 0.00005"
TRAIN="--steps 1e4 --batch_num 16 --batch_size 512 --target_update_tau 5e-2 --learning_starts 100 --gamma 0.99 --buffer_size 1e7 --worker 8"
MODEL="--node 512 --hidden_n 2"
OPTIONS="--initial_eps 0.1 --eps_decay 0"
OPTIMIZER="--optimizer rmsprop"


python run_apex_dpg.py --algo TD3 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
