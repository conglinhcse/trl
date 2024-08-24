CUDA_VISIBLE_DEVICES=0 python ctg/model/sft/sft.py \
    --model_name_or_path="facebook/opt-125m" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --max_seq_length=256 \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=16 \
    --output_dir="ctg/ckpts/sft/sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --gradient_checkpointing

