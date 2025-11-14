#!/bin/bash  
#SBATCH --job-name=infer_clevr  
#SBATCH --output=logs/infer_clevr_%j.out  
#SBATCH --error=logs/infer_clevr_%j.err  
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=4  
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1  
#SBATCH --time=12:00:00  
#SBATCH --mem=32G 

module load Mambaforge/23.11.0-fasrc01
mamba activate IGM

export WANDB_API_KEY="4b9dc043dac87d50c74128f48c230b1755f954ea"


CUDA_VISIBLE_DEVICES=0 python infer_clevr.py \
--learning_rate 4e-3 \
--scheduler_config configs/clevr-2D-pos/scheduler/scheduler_config.json \
--unet_config configs/clevr-2D-pos/unet/config.json \
--dataset_root data/CLEVR_v1.0 --resolution 64
