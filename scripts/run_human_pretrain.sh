
export HF_TOKEN=""
export WANDB_API_KEY=""

torchrun --nproc_per_node=8 --standalone \
    scripts/train.py \
    --config vitra/configs/human_pretrain.json \