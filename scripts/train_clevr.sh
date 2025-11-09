#!/bin/bash  
#SBATCH --job-name=train_clevr  
#SBATCH --output=logs/train_clevr_%j.out  
#SBATCH --error=logs/train_clevr_%j.err  
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=8  
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:2  
#SBATCH --time=48:00:00  
#SBATCH --mem=64G  

module load Mambaforge/23.11.0-fasrc01
mamba activate IGM

export WANDB_API_KEY="4b9dc043dac87d50c74128f48c230b1755f954ea"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes=2 \
--main_process_port 28100 train_clevr.py --enable_xformers_memory_efficient_attention \
--resume_from_checkpoint=latest \
--dataloader_num_workers 4 --learning_rate 2e-5 --mixed_precision fp16 --num_validation_images 10 \
--val_batch_size 10 --max_train_steps 100000 --checkpointing_steps 5000 --checkpoints_total_limit 1 \
--gradient_accumulation_steps 1 --seed 42 \
--output_dir ./outputs/ \
--scheduler_config configs/clevr-2D-pos/scheduler/scheduler_config.json \
--unet_config configs/clevr-2D-pos/unet/config.json \
--dataset_root data/CLEVR_v1.0 --train_batch_size 128 \
--resolution 64 --validation_steps 1000 --tracker_project_name clevr

