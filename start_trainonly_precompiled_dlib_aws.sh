#!/bin/bash

source activate pytorch
# git clone https://github.com/ShivamShrirao/diffusers.git sdw
# cd /home/ubuntu/sdw/examples/dreambooth
# wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py \
#     && wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py \
#     && pip install -qq git+https://github.com/ShivamShrirao/diffusers@219e279b0376d60382fce6a993641f806710ac44  \
#     && pip install -U --pre triton==2.0.0.dev20221105 \
#     && pip install accelerate==0.12.0 transformers==4.24.0 ftfy==6.1.1 bitsandbytes gradio natsort==8.2.0

# export MODEL_ID=$_MODEL_ID
# export STYLE_ID=$_STYLE_ID
# export MODEL_KEY=$_MODEL_KEY
# export MODEL_CLASS=$_MODEL_CLASS
# export MODEL_STEPS=$_MODEL_STEPS
# export MODEL_PATH=$_MODEL_PATH
# export MODEL_BUCKET=$_MODEL_BUCKET
# export REPO_ID=$_REPO_ID
# export BATCH_SAMPLES=$_BATCH_SAMPLES
# export CFG_SCALE=$_CFG_SCALE
# export CONVERT_PRETRAINED_TO_DIFFUSERS=$_CONVERT_PRETRAINED_TO_DIFFUSERS
# export MODEL_NAME=$_PRETRAINED_MODEL_PATH
# export SAMPLING_STEPS=$_SAMPLING_STEPS
# export SAMPLING_METHOD=$_SAMPLING_METHOD
# export RANDOM_PROMPTS=$_RANDOM_PROMPTS
# export WIDTH=$_WIDTH
# export HEIGHT=$_HEIGHT
# export RESTORE_FACES=$_RESTORE_FACES
# export BATCH_ITER=$_BATCH_ITER
# export HIGHRES_FIX=$_HIGHRES_FIX
# export PROMPTLIST=$PROMPTLIST
# # export _HIGHRES_FIX_DENOISING=$_HIGHRES_FIX_DENOISING
# export TG_API_KEY=$_TG_API_KEY
# export TG_CHANNEL_ID=$_TG_CHANNEL_ID

cd /home/ubuntu/sdw
# git reset --hard
# lets use the old commit since new ones are broken. 
# git checkout 219e279b0376d60382fce6a993641f806710ac44

cd ~/
rm -rf imagegen
git clone https://github.com/$REPO_ID/imagegen.git imagegen

if [[ -z ${_STYLE_ID+x} ]]; then 
    
    curl -X POST \
        -H "Content-Type: application/json" \
        -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Starting processing for $MODEL_ID $MODEL_KEY $MODEL_CLASS\", \"disable_notification\": true}" \
        https://api.telegram.org/$TG_API_KEY/sendMessage

else 
    
    curl -X POST \
        -H "Content-Type: application/json" \
        -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Starting processing for $MODEL_ID $MODEL_KEY $MODEL_CLASS with style $STYLE_ID\", \"disable_notification\": true}" \
        https://api.telegram.org/$TG_API_KEY/sendMessage

fi

# pip install awscli --upgrade --user
# mv /root/.local/bin/aws* /bin

# export AWS_ACCESS_KEY_ID=$S3_AK_ID
# export AWS_SECRET_ACCESS_KEY=$S3_AKS
# export AWS_DEFAULT_REGION="ap-southeast-1"
# export AWS_DEFAULT_OUTPUT="json"

aws configure set aws_access_key_id AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key AWS_SECRET_ACCESS_KEY
aws configure set region "ap-southeast-1"
aws configure set output "json"

cd ~
mkdir -p ~/.huggingface
echo -n "$HUGGINGFACE_TOKEN" > ~/.huggingface/token


GPU_NAME=`nvidia-smi --query-gpu=gpu_name --format=csv,noheader`

# aws s3 cp s3://$MODEL_BUCKET/xformerwheels/ ./ --recursive

# aws s3 cp s3://$MODEL_BUCKET/class_images/person /home/ubuntu/sdw/examples/dreambooth/content/data/person --recursive

# aws s3 cp s3://$MODEL_BUCKET/class_images/ /home/ubuntu/sdw/examples/dreambooth/content/data/ --recursive

if [[ $GPU_NAME == *"3090"* ]]; then
    export FORCE_CUDA="1" && \
        export TORCH_CUDA_ARCH_LIST=8.6 && \
        CUDA_VISIBLE_DEVICES=0 pip install ~/xformers-0.0.14.dev0-cp39-cp39-linux_x86_64.whl
fi

if [[ $GPU_NAME == *"A100"* ]]; then    
    export FORCE_CUDA="1" && \
        CUDA_VISIBLE_DEVICES=0 pip install ~/a100/xformers-0.0.15.dev0+fd21b40.d20221115-cp39-cp39-linux_x86_64.a100.cuda11.8.whl
fi

mkdir -p /home/ubuntu/sdw/examples/dreambooth/content/data/$MODEL_ID
cd /home/ubuntu/sdw/examples/dreambooth/content/data/$MODEL_ID/
aws s3 cp s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/$TRAINING_PATH ./training_samples --recursive
if [[ -z ${_STYLE_ID+x} ]]; then 
    echo "var is unset";
else 
    mkdir -p /home/ubuntu/sdw/examples/dreambooth/content/data/$STYLE_ID
    cd /home/ubuntu/sdw/examples/dreambooth/content/data/$STYLE_ID/
    aws s3 cp s3://$MODEL_BUCKET/$MODEL_PATH/$STYLE_ID/ ./training_samples --recursive
fi



cd /home/ubuntu/sdw/examples/dreambooth/content/data/$MODEL_ID/
rm -rf uncropped
mv ./training_samples ./uncropped

python3 ~/imagegen/face_cropper.py /home/ubuntu/sdw/examples/dreambooth/content/data/$MODEL_ID/uncropped /home/ubuntu/sdw/examples/dreambooth/content/data/$MODEL_ID/training_samples


mkdir -p ~/.cache/huggingface/accelerate/

cat <<EOT > ~/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: 'NO'
downcast_bf16: 'no'
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
use_cpu: false
EOT

cd ~/


if [[ $CONVERT_PRETRAINED_TO_DIFFUSERS == 1 ]]; then    
    
    
    # change model name to 
    
    # git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
    git clone https://github.com/CompVis/stable-diffusion.git
    
    # download ckpt from link
    # aws s3 cp s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/$MODEL_ID.ckpt ~/$MODEL_ID.ckpt 
    wget -O ~/$MODEL_ID.ckpt $MODEL_NAME

    # wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_original_stable_diffusion_to_diffusers.py
    # wget -O convert_original_stable_diffusion_to_diffusers.py -q https://raw.githubusercontent.com/cian0/shivamdiffusers/master/scripts/convert_original_stable_diffusion_to_diffusers.py?token=GHSAT0AAAAAAB3NPWQKSKAPG27I4RVRBQX4Y33HGPA
    pip install omegaconf
    pip install boto3 
    
    # convert ckpt referenced model to diffusers
    mv ~/imagegen/convert_original_stable_diffusion_to_diffusers.py ~/convert_original_stable_diffusion_to_diffusers.py
    mv ~/imagegen/v1-inference.yaml ~/v1-inference.yaml
    
    python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "~/$MODEL_ID.ckpt" --original_config_file "~/v1-inference.yaml" --dump_path "~/model"
    export MODEL_NAME="~/model"

fi


cd /home/ubuntu/sdw/

export OUTPUT_DIR="/home/ubuntu/sdw/examples/dreambooth/stable_diffusion_weights/output" 

if [[ -z ${_STYLE_ID+x} ]]; then 
    cat <<EOT > /home/ubuntu/sdw/examples/dreambooth/concepts_list.json
    [{
        "instance_prompt":      "photo of $MODEL_ID person",
        "class_prompt":         "photo of a person",
        "instance_data_dir":    "content/data/$MODEL_ID/training_samples/",
        "class_data_dir":       "content/data/person"
    }]
EOT
else 
    cat <<EOT > /home/ubuntu/sdw/examples/dreambooth/concepts_list.json
    [{
        "instance_prompt":      "photo of $MODEL_ID person",
        "class_prompt":         "photo of a person",
        "instance_data_dir":    "content/data/$MODEL_ID/training_samples/",
        "class_data_dir":       "content/data/person"
    },{
        "instance_prompt":      "$STYLE_ID artstyle",
        "class_prompt":         "artstyle",
        "instance_data_dir":    "content/data/$STYLE_ID/training_samples/",
        "class_data_dir":       "content/data/artstyle"
    }]
EOT
fi

cd /home/ubuntu/sdw/examples/dreambooth/

if [[ -z ${_STYLE_ID+x} ]]; then 
    # TRAINING_FILE_COUNT=`ls /etc | wc -l`*100
    STEPS_BASED_ON_FILES=$((`ls /home/ubuntu/sdw/examples/dreambooth/content/data/$MODEL_ID/training_samples | wc -l`*110))

    if [[ $STEPS_BASED_ON_FILES < 2200 ]]; then    
        STEPS_BASED_ON_FILES=2200 #minimum number of steps
    fi
else 
    # TRAINING_FILE_COUNT=`ls /etc | wc -l`*100
    STEPS_BASED_ON_FILES=$((`ls /home/ubuntu/sdw/examples/dreambooth/content/data/$MODEL_ID/training_samples | wc -l`*110))

    if [[ $STEPS_BASED_ON_FILES < 4400 ]]; then    
        STEPS_BASED_ON_FILES=4400 #minimum number of steps
    fi
fi

curl -X POST \
     -H "Content-Type: application/json" \
     -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Starting training for $MODEL_ID $MODEL_KEY $MODEL_CLASS with $STEPS_BASED_ON_FILES steps\", \"disable_notification\": true}" \
     https://api.telegram.org/$TG_API_KEY/sendMessage


export GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | tee /dev/tty)

touch ~/logs_training

export OUTPUT_DIR_TRAINING="stable_diffusion_weights/output" 

if [[ $GPU_NAME == *"NVIDIA A10G"* ]]; then
    pip uninstall -y xformers

    accelerate launch train_dreambooth.py \
        --pretrained_model_name_or_path="$MODEL_NAME" \
        --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
        --output_dir=$OUTPUT_DIR_TRAINING \
        --revision="fp16" \
        --seed=1337 \
        --resolution=512 \
        --train_batch_size=1 \
        --train_text_encoder \
        --center_crop \
        --mixed_precision="fp16" \
        --gradient_accumulation_steps=1 \
        --learning_rate=1e-6 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --sample_batch_size=4 \
        --max_train_steps=$STEPS_BASED_ON_FILES \
        --save_interval=500 \
        --save_sample_prompt="photo of $MODEL_ID person" \
        --concepts_list="concepts_list.json" 

        # --with_prior_preservation --prior_loss_weight=1.0 \
        # --num_class_images=80 \
        # --gradient_checkpointing \
        # --use_8bit_adam \?
        # --gradient_checkpointing \
        # --not_cache_latents \
fi > ~/logs_training

if [[ $GPU_NAME == *"Tesla T4"* ]]; then
    accelerate launch train_dreambooth.py \
        --pretrained_model_name_or_path="$MODEL_NAME" \
        --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
        --output_dir=$OUTPUT_DIR_TRAINING \
        --revision="fp16" \
        --with_prior_preservation --prior_loss_weight=1.0 \
        --seed=1337 \
        --resolution=512 \
        --train_batch_size=1 \
        --gradient_checkpointing \
        --use_8bit_adam \
        --train_text_encoder \
        --center_crop \
        --mixed_precision="fp16" \
        --gradient_accumulation_steps=2 \
        --learning_rate=1e-6 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=80 \
        --sample_batch_size=4 \
        --max_train_steps=$STEPS_BASED_ON_FILES \
        --save_interval=$STEPS_BASED_ON_FILES \
        --save_sample_prompt="photo of $MODEL_ID person" \
        --concepts_list="concepts_list.json" 

        # --not_cache_latents \
fi > ~/logs_training

if [[ $GPU_NAME == *"3090"* ]]; then
    accelerate launch train_dreambooth.py \
        --pretrained_model_name_or_path="$MODEL_NAME" \
        --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
        --output_dir=$OUTPUT_DIR_TRAINING \
        --revision="fp16" \
        --with_prior_preservation --prior_loss_weight=1.0 \
        --seed=1337 \
        --resolution=512 \
        --train_batch_size=1 \
        --train_text_encoder \
        --center_crop \
        --mixed_precision="fp16" \
        --gradient_accumulation_steps=2 \
        --learning_rate=1e-6 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=80 \
        --sample_batch_size=4 \
        --max_train_steps=$STEPS_BASED_ON_FILES \
        --save_interval=$STEPS_BASED_ON_FILES \
        --save_sample_prompt="photo of $MODEL_ID person" \
        --concepts_list="concepts_list.json" 
fi > ~/logs_training

if [[ $GPU_NAME == *"A100"* ]]; then    
    accelerate launch train_dreambooth.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
        --output_dir=$OUTPUT_DIR_TRAINING \
        --revision="fp16" \
        --with_prior_preservation --prior_loss_weight=1.0 \
        --seed=1337 \
        --resolution=512 \
        --train_batch_size=24 \ #change to 8 if 40GB alloc
        --train_text_encoder \
        --center_crop \
        --mixed_precision="fp16" \
        --gradient_accumulation_steps=2 \
        --learning_rate=1e-6 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=80 \
        --sample_batch_size=4 \
        --max_train_steps=$STEPS_BASED_ON_FILES \
        --save_interval=$STEPS_BASED_ON_FILES \
        --save_sample_prompt="photo of $MODEL_ID person" \
        --concepts_list="concepts_list.json"
fi

curl -X POST \
     -H "Content-Type: application/json" \
     -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Finished training for $MODEL_ID $MODEL_KEY $MODEL_CLASS\", \"disable_notification\": true}" \
     https://api.telegram.org/$TG_API_KEY/sendMessage
     
mv ~/imagegen/convert_diffusers_to_original_stable_diffusion.py ./convert_diffusers_to_original_stable_diffusion.py
python convert_diffusers_to_original_stable_diffusion.py --model_path "$OUTPUT_DIR/$STEPS_BASED_ON_FILES"  --checkpoint_path "$OUTPUT_DIR/$STEPS_BASED_ON_FILES/model.ckpt" --half

aws s3 cp "$OUTPUT_DIR/$STEPS_BASED_ON_FILES/model.ckpt" s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/$MODEL_ID.ckpt
aws s3 cp "~/logs" s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/logs.txt

curl -X POST \
     -H "Content-Type: application/json" \
     -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Done uploading model for $MODEL_ID $MODEL_KEY $MODEL_CLASS\", \"disable_notification\": true}" \
     https://api.telegram.org/$TG_API_KEY/sendMessage

# end instance
# cd ~
# cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key;
# apt-get install -y wget; wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;

FILE=$OUTPUT_DIR/$STEPS_BASED_ON_FILES/model.ckpt
if test -f "$FILE"; then
    echo "$FILE exists."
    # ./vast destroy instance ${VAST_CONTAINERLABEL:2}
else 
    curl -X POST \
        -H "Content-Type: application/json" \
        -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Model cannot be uploaded for $MODEL_ID $MODEL_KEY $MODEL_CLASS, not shutting down\", \"disable_notification\": true}" \
        https://api.telegram.org/$TG_API_KEY/sendMessage
fi

sleep infinity
wait