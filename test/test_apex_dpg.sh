export CUDA_VISIBLE_DEVICES=1
export DISPLAY=:0

pip install -q ..

#ENV="--env BipedalWalker-v3"
ENV="--env Walker2d-v4"
#"Walker2d-v4"
RL="--learning_rate 0.0003"
TRAIN="--steps 2e6 --batch 256 --target_update_tau 5e-3 --learning_starts 100 --gamma 0.99 --buffer_size 1e6 --worker 4"
MODEL="--node 512 --hidden_n 2"
OPTIONS="--initial_eps 0.1 --eps_decay 0 --n_step 3"
OPTIMIZER="--optimizer adam"

python run_apex_dpg.py --algo TD3 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
