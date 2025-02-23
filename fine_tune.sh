export HF_ENDPOINT=https://hf-mirror.com

export CUDA_VISIBLE_DEVICES=7
CUDA_VISIBLE_DEVICES="0" accelerate launch train.py \
    --config configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml \
    --dataset-dir "Emilia-101k" \
    --run-name whsp_ft \
    --batch-size 2 \
    --max-steps 400000 \
    --max-epochs 1000 \
    --save-every 100 \
    --num-workers 16