export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=dummy
python run_pg.py --algo PPO --env BreakoutNoFrameskip-v4 --steps 2e6 --worker 8 --batch 128 --mini_batch 256 --node 256 --hidden_n 1 --gamma 0.995 --lamda 0.995 --ent_coef 0.01 --gae_normalize --optimizer adam
python run_pg.py --algo A2C --env BreakoutNoFrameskip-v4 --steps 2e6 --worker 8 --batch 32 --node 512 --hidden_n 1 --ent_coef 5e-2 --optimizer adam