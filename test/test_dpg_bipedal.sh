export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=dummy

ENV="--env BipedalWalker-v3"
RL="--learning_rate 0.0002"
TRAIN="--steps 2e7 --batch 256 --target_update_tau 5e-4 --learning_starts 1000"
MODEL="--node 512 --hidden_n 3"
OPTIONS=""
OPTIMIZER="--optimizer adamw"
python  run_dpg.py --algo TQC $ENV $RL $TRAIN $MODEL $OPTIMIZER $OPTIONS