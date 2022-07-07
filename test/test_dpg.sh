export CUDA_VISIBLE_DEVICES=1
export SDL_VIDEODRIVER=dummy

#python run_dpg.py --env ../../env/Walker.x86_64 --algo SAC --batch 128 --target_update_tau 1e-4 --learning_starts 50000 --step 5e6 --node 512 --hidden_n 3 &
#python run_dpg.py --env ../../env/Walker.x86_64 --algo TQC --batch 128 --target_update_tau 1e-4 --learning_starts 50000 --step 5e6 --node 512 --hidden_n 3 --quantile_drop 0.05 --worker_id 2
#python run_dpg.py --env Pendulum-v1 --algo TQC --batch 128 --target_update_tau 1e-3 --learning_starts 2000 --step 1e5 --node 512 --hidden_n 2
#python run_dpg.py --env Pendulum-v1 --algo IQA --batch 128 --target_update_tau 1e-3 --learning_starts 100 --step 5e5 --node 512 --hidden_n 2

#python run_dpg.py --env ../../env/Walker.x86_64 --algo SAC --batch 128 --target_update_tau 5e-3 --learning_starts 1000 --step 5e6 --node 512 --hidden_n 3 &
#python run_dpg.py --env ../../env/Walker.x86_64 --algo TQC --batch 128 --target_update_tau 5e-3 --learning_starts 1000 --step 5e6 --node 512 --hidden_n 3 --worker_id 2

#python run_dpg.py --env ../../env/Walker.x86_64 --algo  --batch 32 --target_update_tau 2e-4 --learning_starts 50000 --step 5e6 --node 512 --hidden_n 3 --risk_avoidance 1.0 --worker_id 2
#python run_dpg.py --env ../../env/Walker.x86_64 --algo IQA --batch 128 --target_update_tau 1e-4 --learning_starts 50000 --step 5e6 --node 512 --hidden_n 3

#python run_dpg.py --env Humanoid-v3 --algo TD4_QR --batch 32 --learning_starts 50000 --step 2e7 --node 512 --hidden_n 3 --action_noise 0.1 --risk_avoidance 0.0 & #truncated 
#python run_dpg.py --env Humanoid-v3 --algo TD4_QR --batch 32 --learning_starts 50000 --step 2e7 --node 512 --hidden_n 3 --action_noise 0.05 --n_support 25 --mixture truncated --risk_avoidance 0.8 &

#python run_dpg.py --env Humanoid-v3 --algo SAC --batch 32 --learning_starts 50000 --step 2e7 --node 512 --hidden_n 3
#python run_dpg.py --env Humanoid-v3 --algo SAC --batch 32 --target_update_tau 5e-4 --learning_starts 1000 --step 2e7 --node 512 --hidden_n 3 &
#python run_dpg.py --env Humanoid-v3 --algo TQC --batch 32 --target_update_tau 5e-4 --learning_starts 1000 --step 2e7 --node 512 --hidden_n 3 &

export CUDA_VISIBLE_DEVICES=0
#python run_dpg.py --env BipedalWalkerHardcore-v3 --algo TQC --batch 128 --learning_starts 50000 --step 2e7 --node 512 --hidden_n 3 --risk_avoidance 0.0 --quantile_drop 0.1
#python run_dpg.py --env ../../env/Walker.x86_64 --algo SAC --step 2e6 --node 512 --hidden_n 3 --optimizer adam --worker_id 2 &
#python run_dpg.py --env ../../env/Walker.x86_64 --algo TQC --step 2e6 --node 512 --hidden_n 3 --optimizer adam --worker_id 3 --mixture truncated

python run_dpg.py --env Humanoid-v3 --algo IQA --batch 256 --target_update_tau 5e-4 --learning_starts 1000 --step 2e7 --node 512 --hidden_n 3
#python run_dpg.py --env Humanoid-v3 --algo TD4_QR --batch 256 --target_update_tau 5e-4 --learning_starts 1000 --step 2e7 --node 512 --hidden_n 3
