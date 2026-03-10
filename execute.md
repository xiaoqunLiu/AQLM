CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python main.py \
    Qwen/Qwen2.5-7B-Instruct \
    c4 \
    --save ./qwen2.5-7b-instruct-aqlm-2bit \
    --nbits_per_codebook 16 \
    --in_group_size 8 \
    --num_codebooks 1 \
    --relative_mse_tolerance 0.01 \
    --finetune_batch_size 16 \
    --finetune_max_epochs 10 \
    --finetune_early_stop 3 \
    --finetune_keep_best \
    --local_batch_size 1 \
    --nsamples 1024 \
    --val_size 128 \
    --attn_implementation sdpa \
    --trust_remote_code \
    > nohup.out 2>&1 &

CUDA_VISIBLE_DEVICES=1,2 nohup python main.py \
    Qwen/Qwen2.5-0.5B-Instruct \
    c4 \
    --save ./qwen2.5-0.5b-instruct-aqlm-2bit \
    --nbits_per_codebook 16 \
    --in_group_size 8 \
    --num_codebooks 1 \
    --relative_mse_tolerance 0.01 \
    --finetune_batch_size 16 \
    --finetune_max_epochs 10 \
    --finetune_early_stop 3 \
    --finetune_keep_best \
    --local_batch_size 1 \
    --nsamples 1024 \
    --val_size 128 \
    --attn_implementation sdpa \
    --trust_remote_code \
    > nohup_qwen2.5_0.5b.out 2>&1 &
