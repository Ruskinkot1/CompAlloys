#!/bin/bash
#SBATCH --job-name=ocp_hea_gpu
#SBATCH --output=/home/user/CompAlloys/logs/ocp_hea_gpu.log
#SBATCH --partition=local
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00

# 激活虚拟环境（环境内已安装 CUDA/cuDNN）
source /home/user/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc  # 加载 Shell 配置
conda activate HEA

# 运行代码
python /home/user/CompAlloys/ocp_calculatror-JW.py