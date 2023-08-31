export CUDA_VISIBLE_DEVICES=1
export DISPLAY=:0

pip install -q ..

ENV="--env ../../env/Walker.x86_64"
RL="--learning_rate 0.00005"
TRAIN="--steps 1e6 --buffer_size 1e6 --batch 64 --target_update_tau 1e-3 --learning_starts 5000 --gradient_steps 10"
MODEL="--node 512 --hidden_n 3"
OPTIONS="--n_support 25 --time_scale 20 --mixture truncated --quantile_drop 0.05 --cvar 0.3 --capture_frame_rate 30"
OPTIMIZER="--optimizer adam"
python  run_dpg.py --algo TD3 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo SAC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TD4_QR $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TQC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TD4_IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TQC_IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
