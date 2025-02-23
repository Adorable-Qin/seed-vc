export HF_ENDPOINT=https://hf-mirror.com

export CUDA_VISIBLE_DEVICES=7
python inference.py \
    --source "/mnt/workspace/home/fangzihao/seed-vc/sample/whisper2.wav" \
    --target "/mnt/workspace/home/fangzihao/seed-vc/sample/target.wav" \
    --output /mnt/workspace/home/fangzihao/seed-vc/sample \
    --diffusion-steps 25 \
    --length-adjust 1.0 \
    --inference-cfg-rate 0.7 \
    --f0-condition False \
    --auto-f0-adjust False \
    --semi-tone-shift 0 \
    --fp16 True \
    --checkpoint /mnt/workspace/home/fangzihao/seed-vc/checkpoints/whsp_ft/DiT_epoch_00000_step_12600.pth \
    --config /mnt/workspace/home/fangzihao/seed-vc/runs/whsp_ft/config_dit_mel_seed_uvit_whisper_small_wavenet.yml

    # --source "/mnt/workspace/home/fangzihao/seed-vc/sample/whisper3.wav" /mnt/workspace/home/fangzihao/WHSP_LGU/parallel/user013/task001/whsp.wav
    # --target "/mnt/workspace/home/fangzihao/seed-vc/sample/target3.wav" \/mnt/workspace/home/fangzihao/WHSP_LGU/parallel/user013/task003/norm.wav