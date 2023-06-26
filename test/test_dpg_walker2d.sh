export CUDA_VISIBLE_DEVICES=1
export DISPLAY=:0

pip install -q ..

ENV="--env Walker2d-v4"
RL="--learning_rate 0.0003"
TRAIN="--steps 2e6 --buffer_size 1e6 --batch 256 --target_update_tau 1e-3 --learning_starts 50000"
MODEL="--node 512 --hidden_n 2"
OPTIONS=""
OPTIMIZER="--optimizer adam"

#python  run_dpg.py --algo SAC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

OPTIONS="--per"

python  run_dpg.py --algo SAC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS