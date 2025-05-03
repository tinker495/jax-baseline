export CUDA_VISIBLE_DEVICES=2
export DISPLAY=:0

pip install -q ..

ENV="--env Walker2d-v4"
LR="--learning_rate 0.0003"
TRAIN="--steps 2e6 --buffer_size 1e6 --batch 256 --target_update_tau 1e-3 --learning_starts 50000"
MODEL="--node 512 --hidden_n 2"
OPTIONS=""
OPTIMIZER="--optimizer adam"

python  run_dpg.py --algo TD3 $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
