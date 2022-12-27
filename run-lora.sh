#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_KEY="asian_men_man_tutor_korean_picture_young"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/ian/projs/lora/training_images"
export OUTPUT_DIR="/home/ian/projs/lora/outputs/$INSTANCE_KEY""_2k_v1_5"
export CLASS_DIR="/home/ian/projs/lora/classes_1_5/person"

accelerate launch train_lora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --instance_prompt="photo of $INSTANCE_KEY person" \
  --class_prompt="photo of a person" \
  --train_text_encoder \
  --resolution=512 \
  --with_prior_preservation \
  --train_batch_size=1 \
  --color_jitter \
  --mixed_precision="fp16" \
  --revision="fp16" \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-6 \
  --lr_scheduler="linear" \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000


  # --train_batch_size=6 \ # max

  # --pretrained_model_name_or_path "/home/ian/projs/lora/sd-vae-ft-mse" \



  #https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_KEY="asian_men_man_tutor_korean_picture_young"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/ian/projs/lora/training_images"
export OUTPUT_DIR="/home/ian/projs/lora/outputs/$INSTANCE_KEY""_2200_v1_5"
export CLASS_DIR="/home/ian/projs/lora/classes_1_5/person"

accelerate launch train_lora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --instance_prompt="photo of $INSTANCE_KEY person" \
  --class_prompt="photo of a person" \
  --train_text_encoder \
  --resolution=512 \
  --with_prior_preservation \
  --train_batch_size=6 \
  --color_jitter \
  --mixed_precision="fp16" \
  --revision="fp16" \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-6 \
  --lr_scheduler="linear" \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --lr_warmup_steps=0 \
  --max_train_steps=2200



  

export INSTANCE_KEY="man_men_portrait_white_frames_handsome_black"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/ubuntu/lora/training_images"
export OUTPUT_DIR="/home/ubuntu/lora/outputs/$INSTANCE_KEY""_2k_v1_5"
export CLASS_DIR="/home/ubuntu/lora/classes_1_5/person"

accelerate launch train_lora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --instance_prompt="photo of $INSTANCE_KEY person" \
  --class_prompt="photo of a person" \
  --train_text_encoder \
  --resolution=512 \
  --with_prior_preservation \
  --train_batch_size=2 \
  --color_jitter \
  --mixed_precision="fp16" \
  --revision="fp16" \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-6 \
  --lr_scheduler="linear" \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --lr_warmup_steps=0 \
  --max_train_steps=2200




export INSTANCE_KEY="royalty_korean_andrew_ryan_lee_nguyen_joseph"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/ian/projs/lora/training_images"
export OUTPUT_DIR="/home/ian/projs/lora/outputs/$INSTANCE_KEY""_1100_v1_5"
export CLASS_DIR="/home/ian/projs/lora/classes_1_5/person"

accelerate launch train_lora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --instance_prompt="photo of $INSTANCE_KEY person" \
  --class_prompt="photo of a person" \
  --train_text_encoder \
  --resolution=512 \
  --with_prior_preservation \
  --train_batch_size=8 \
  --color_jitter \
  --mixed_precision="fp16" \
  --revision="fp16" \
  --learning_rate=4e-4 \
  --learning_rate_text=5e-6 \
  --lr_scheduler="constant" \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --lr_warmup_steps=0 \
  --max_train_steps=1100


  # --gradient_accumulation_steps=1 \
  # --gradient_checkpointing \


  #https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export INSTANCE_KEY="royalty_korean_andrew_ryan_lee_nguyen_joseph"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/ian/projs/lora/training_images/klao"
export OUTPUT_DIR="/home/ian/projs/lora/outputs/$INSTANCE_KEY""_1000_pt_v1_5"
export CLASS_DIR="/home/ian/projs/lora/classes_1_5/person"

accelerate launch train_lora_w_ti.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=20 \
  --gradient_accumulation_steps=1 \
  --learning_rate=4e-4 \
  --learning_rate_text=5e-6 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=100 \
  --max_train_steps=1000 \
  --placeholder_token="$INSTANCE_KEY" \
  --learnable_property="object"\
  --initializer_token="man" \
  --save_steps=500 \
  --unfreeze_lora_step=500



  #https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export INSTANCE_KEY="olrak_3_lao"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/ian/projs/lora/training_images/klao"
export OUTPUT_DIR="/home/ian/projs/lora/outputs/$INSTANCE_KEY""_1500_pt_v1_5"
export CLASS_DIR="/home/ian/projs/lora/classes_1_5/person"

accelerate launch train_lora_w_ti.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=20 \
  --gradient_accumulation_steps=1 \
  --learning_rate=4e-4 \
  --learning_rate_text=5e-6 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=100 \
  --max_train_steps=1500 \
  --placeholder_token="$INSTANCE_KEY" \
  --learnable_property="object"\
  --initializer_token="man" \
  --save_steps=500 \
  --unfreeze_lora_step=750



#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export INSTANCE_KEY="<olrk_>"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/ian/projs/lora/training_images/klao"
export OUTPUT_DIR="/home/ian/projs/lora/outputs/$INSTANCE_KEY""_3000_pt_v1_5"
export CLASS_DIR="/home/ian/projs/lora/classes_1_5/person"


accelerate launch train_lora_w_ti.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-6 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=100 \
  --max_train_steps=6000 \
  --placeholder_token="$INSTANCE_KEY" \
  --learnable_property="object"\
  --initializer_token="man" \
  --save_steps=500 \
  --unfreeze_lora_step=4000 



#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export INSTANCE_KEY="<olrk_2>"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/ian/projs/lora/training_images/klao"
export OUTPUT_DIR="/home/ian/projs/lora/outputs/$INSTANCE_KEY""_3000_pt_v1_5"
export CLASS_DIR="/home/ian/projs/lora/classes_1_5/person"


accelerate launch train_lora_w_ti.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-6 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --max_train_steps=3000 \
  --placeholder_token="$INSTANCE_KEY" \
  --learnable_property="object"\
  --initializer_token="man" \
  --save_steps=500 \
  --unfreeze_lora_step=2000 



#high LR
export INSTANCE_KEY="olrk_2"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/ian/projs/lora/training_images/klao"
export OUTPUT_DIR="/home/ian/projs/lora/outputs/$INSTANCE_KEY""_3000_pt_v1_5"
export CLASS_DIR="/home/ian/projs/lora/classes_1_5/person"


accelerate launch train_lora_w_ti.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-3 \
  --learning_rate_text=5e-5 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --max_train_steps=3000 \
  --placeholder_token="$INSTANCE_KEY" \
  --learnable_property="object"\
  --initializer_token="man" \
  --save_steps=500 \
  --unfreeze_lora_step=1500 


  # --stochastic_attribute "portrait of olrk_2" \




#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export INSTANCE_KEY="<olrk_2>"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/ian/projs/lora/training_images/klao"
export OUTPUT_DIR="/home/ian/projs/lora/outputs/$INSTANCE_KEY""_3000_pt_v1_5"
export CLASS_DIR="/home/ian/projs/lora/classes_1_5/person"


accelerate launch train_lora_w_ti.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --stochastic_attribute "person" \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-6 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --max_train_steps=3000 \
  --placeholder_token="$INSTANCE_KEY" \
  --learnable_property="object"\
  --initializer_token="man" \
  --save_steps=500 \
  --unfreeze_lora_step=2000 
  


export INSTANCE_KEY="1k_lao"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/ian/projs/lora/training_images/klao"
export OUTPUT_DIR="/home/ian/projs/lora/outputs/$INSTANCE_KEY""_1000_v1_5"

lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate_unet=1e-4 \
  --learning_rate_text=1e-5 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --placeholder_token="$INSTANCE_KEY" \
  --learnable_property="object"\
  --initializer_token="person" \
  --save_steps=500 \
  --max_train_steps_ti=2500 \
  --max_train_steps_tuning=1000 \
  --perform_inversion=True \
  --weight_decay_ti=0.1 \
  --weight_decay_lora=0.1\
  --device="cuda:0"




export INSTANCE_KEY="mohamed_moe"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/ian/projs/lora/training_images/mohamed_moe"
export OUTPUT_DIR="/home/ian/projs/lora/outputs/$INSTANCE_KEY""_latest_8000_v1_5"

lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate_unet=1e-4 \
  --learning_rate_text=1e-5 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --placeholder_token="$INSTANCE_KEY" \
  --learnable_property="object"\
  --initializer_token="person" \
  --save_steps=500 \
  --max_train_steps_ti=2500 \
  --max_train_steps_tuning=8000 \
  --perform_inversion=True \
  --weight_decay_ti=0.1 \
  --weight_decay_lora=0.1\
  --device="cuda:0"



#cosine can be linear