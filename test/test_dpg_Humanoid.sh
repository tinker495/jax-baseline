export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env Humanoid-v4"
RL="--learning_rate 0.0003"
TRAIN="--steps 1e7 --buffer_size 1e6 --batch 256 --target_update_tau 5e-3 --learning_starts 50000"
MODEL="--node 256 --hidden_n 2"
OPTIONS=""
OPTIMIZER="--optimizer adam"

export CUDA_VISIBLE_DEVICES=1
python  run_dpg.py --algo TD7 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --seed 128 &
export CUDA_VISIBLE_DEVICES=2
python  run_dpg.py --algo TD7 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --seed 32  &
export CUDA_VISIBLE_DEVICES=3
python  run_dpg.py --algo TD7 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS --seed 512

#export CUDA_VISIBLE_DEVICES=1
#python  run_dpg.py --algo TD3 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS &

#export CUDA_VISIBLE_DEVICES=2
#python  run_dpg.py --algo TQC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

#export CUDA_VISIBLE_DEVICES=3
#python  run_dpg.py --algo TD3 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS

#python  run_dpg.py --algo TD3 $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS &
#python  run_dpg.py --algo SAC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS &
#python  run_dpg.py --algo TD4_QR $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS &
#python  run_dpg.py --algo TQC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS &

#export CUDA_VISIBLE_DEVICES=1

#python  run_dpg.py --algo TD4_IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS &
#python  run_dpg.py --algo TQC_IQN $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS
