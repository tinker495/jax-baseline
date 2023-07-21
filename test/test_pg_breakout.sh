export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

pip install -q ..

ENV="--env BreakoutNoFrameskip-v4"
RL="--learning_rate 0.00003"
TRAIN="--steps 1e6 --worker 32 --batch 32 --mini_batch 256 --gamma 0.995 --lamda 0.95 --ent_coef 0"
MODEL="--node 512 --hidden_n 1"
OPTIONS=""
OPTIMIZER="--optimizer rmsprop"

for i in {1..5}
do
    TRAIN="--steps 1e6 --worker 32 --batch 32 --mini_batch 256 --gamma 0.995 --lamda 0.95 --ent_coef 0"

    python run_pg.py --algo PPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
    python run_pg.py --algo TPPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER

    TRAIN="--steps 1e6 --worker 32 --batch 32 --mini_batch 256 --gamma 0.995 --lamda 0.95 --ent_coef 1e-4"

    python run_pg.py --algo PPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
    python run_pg.py --algo TPPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER

    TRAIN="--steps 1e6 --worker 32 --batch 32 --mini_batch 256 --gamma 0.995 --lamda 0.95 --ent_coef 1e-3"

    python run_pg.py --algo PPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
    python run_pg.py --algo TPPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER

    TRAIN="--steps 1e6 --worker 32 --batch 32 --mini_batch 256 --gamma 0.995 --lamda 0.95 --ent_coef 1e-2"

    python run_pg.py --algo PPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
    python run_pg.py --algo TPPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER

    TRAIN="--steps 1e6 --worker 32 --batch 32 --mini_batch 256 --gamma 0.995 --lamda 0.95 --ent_coef 1e-1"

    python run_pg.py --algo PPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
    python run_pg.py --algo TPPO $ENV $TRAIN $MODEL $OPTIONS $OPTIMIZER
done
