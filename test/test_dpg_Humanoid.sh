export CUDA_VISIBLE_DEVICES=0
apt install xvfb -y -qq

pip install -q ..

EXPERIMENT="--experiment_name DPG_Humanoid"
ENV="--env Humanoid-v4"
LR="--learning_rate 0.0003"
TRAIN="--steps 5e6 --buffer_size 1e6 --batch 256 --target_update_tau 5e-3 --learning_starts 50000"
MODEL="--node 256 --hidden_n 2"
OPTIONS="--logdir log/"
OPTIMIZER="--optimizer adam_low_b1"

#xvfb-run -a python  run_dpg.py --algo CrossQ $EXPERIMENT $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
#xvfb-run -a python  run_dpg.py --algo CrossQ $EXPERIMENT $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --simba
#xvfb-run -a python  run_dpg.py --algo SAC $EXPERIMENT $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
#xvfb-run -a python  run_dpg.py --algo TQC $EXPERIMENT $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
#xvfb-run -a python  run_dpg.py --algo TD3 $EXPERIMENT $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --simba
#xvfb-run -a python  run_dpg.py --algo TD3 $EXPERIMENT $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
#xvfb-run -a python  run_dpg.py --algo SAC $EXPERIMENT $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --simba
#xvfb-run -a python  run_dpg.py --algo SAC $EXPERIMENT $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
xvfb-run -a python  run_dpg.py --algo TQC $EXPERIMENT $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --simba
#xvfb-run -a python  run_dpg.py --algo TQC $EXPERIMENT $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
#xvfb-run -a python  run_dpg.py --algo TD7 $EXPERIMENT $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS --simba
#xvfb-run -a python  run_dpg.py --algo TD7 $EXPERIMENT $ENV $LR $TRAIN $MODEL $OPTIMIZER $OPTIONS
