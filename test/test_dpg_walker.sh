export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=dummy

ENV="--env ../../env/Walker.x86_64"
RL="--learning_rate 0.0002"
TRAIN="--steps 2e6 --batch 256 --target_update_tau 5e-4 --learning_starts 1000"
MODEL="--node 512 --hidden_n 3"
OPTIONS="--time_scale 8"
OPTIMIZER="--optimizer adam"
python  run_dpg.py --algo TD3 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TD4_QR $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TD4_IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo SAC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
python  run_dpg.py --algo TQC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS