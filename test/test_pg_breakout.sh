export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=dummy
python run_pg.py --algo A2C --env BreakoutNoFrameskip-v4 --worker 8 --steps 1e7 --batch 32 --node 512 --hidden_n 1 --ent_coef 5e-2
python run_pg.py --algo PPO --env BreakoutNoFrameskip-v4 --worker 8 --steps 1e7 --batch 128 --node 512 --hidden_n 1 --gamma 0.995 --lamda 0.995 --ent_coef 1e-3 --no_gae_normalize