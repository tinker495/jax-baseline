export CUDA_VISIBLE_DEVICES=1
export SDL_VIDEODRIVER=dummy

ENV="--env Humanoid-v4"
RL="--learning_rate 0.00005"
TRAIN="--steps 5e6 --buffer_size 1e6 --batch 32 --target_update_tau 1e-3 --learning_starts 50000"
MODEL="--node 512 --hidden_n 3"
OPTIONS="--n_support 50 --time_scale 20 --mixture truncated --quantile_drop 0.02"
OPTIMIZER="--optimizer adam"
python  run_dpg.py --algo TD3 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo SAC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TD4_QR $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python  run_dpg.py --algo TD4_IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TQC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS