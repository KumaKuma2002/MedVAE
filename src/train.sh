export TRAIN_DATA_DIR="/mnt/data/jliu452/Data/Dataset901_SMILE/h5" 



accelerate launch --num_processes=1 train_klvae.py \
  --train_data_dir=$TRAIN_DATA_DIR \
  --resume_from_checkpoint="../ckpt/MedVAE_KL-sharp_plain" \
  --validation_images /mnt/data/jliu452/Data/Dataset901_SMILE/h5/baichaoxiao20240416_arterial/ct.h5 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=0 \
  --report_to="wandb" \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --max_train_steps=1_000_000 \
  --vae_loss="l1" \
  --learning_rate=1e-4 \
  --validation_steps=1000 \
  --checkpointing_steps=1000 \
  --checkpoints_total_limit=5 \
  --kl_weight=1e-8 \
  --output_dir="../outputs/klvae"\
  --seg_model_path="../ckpt/segmenter/nnUNetTrainer__nnUNetResEncUNetLPlans__2d" \
  --seg_loss_weight=1e-3\
