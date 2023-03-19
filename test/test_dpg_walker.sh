export CUDA_VISIBLE_DEVICES=1
export SDL_VIDEODRIVER=dummy

ENV="--env ../../env/Walker.x86_64"
RL="--learning_rate 0.0002"
TRAIN="--steps 1e6 --batch 256 --target_update_tau 1e-3 --learning_starts 1000"
MODEL="--node 256 --hidden_n 3"
OPTIONS="--n_support 50 --time_scale 20 --mixture truncated --quantile_drop 0.5"
OPTIMIZER="--optimizer adam"
#python  run_dpg.py --algo TD3 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python  run_dpg.py --algo SAC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TD4_QR $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python  run_dpg.py --algo TD4_IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TQC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

OPTIONS="--n_support 50 --time_scale 20 --mixture truncated --quantile_drop 0.2"
#python  run_dpg.py --algo TD3 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python  run_dpg.py --algo SAC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TD4_QR $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python  run_dpg.py --algo TD4_IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TQC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

OPTIONS="--n_support 50 --time_scale 20 --mixture truncated --quantile_drop 0.1"
#python  run_dpg.py --algo TD3 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python  run_dpg.py --algo SAC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TD4_QR $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python  run_dpg.py --algo TD4_IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TQC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

OPTIONS="--n_support 50 --time_scale 20 --mixture truncated --quantile_drop 0.05"
#python  run_dpg.py --algo TD3 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python  run_dpg.py --algo SAC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TD4_QR $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python  run_dpg.py --algo TD4_IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TQC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

OPTIONS="--n_support 50 --time_scale 20 --mixture truncated --quantile_drop 0.02"
#python  run_dpg.py --algo TD3 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python  run_dpg.py --algo SAC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TD4_QR $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python  run_dpg.py --algo TD4_IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TQC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

OPTIONS="--n_support 50 --time_scale 20 --mixture min"
#python  run_dpg.py --algo TD3 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python  run_dpg.py --algo SAC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TD4_QR $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
#python  run_dpg.py --algo TD4_IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TQC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS