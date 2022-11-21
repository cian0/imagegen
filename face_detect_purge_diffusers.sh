#!/bin/bash
mkdir -p /workspace/matcher/similar/outputs

cd /workspace/matcher

# for getting similar faces 
rm similar/outputs/*
rm recognized.txt
rm recognized_purged.txt
rm recognized_purged2.txt

# user_img_path = '/workspace/cropped/'
# match_output_path = '/workspace/matched/'
# rejected_output_path = '/workspace/rejected/'
# compare_path = '/workspace/stable-diffusion-webui/outputs/txt2img-images/'

face_recognition /workspace/cropped/ /workspace/imagegen/outputs/ --cpus -1 --tolerance 0.55 > recognized.txt
sed '/unknown_person/d;/no_persons_found/d;s/png,.*/png/' recognized.txt > recognized_purged.txt
awk '!seen[$0]++' recognized_purged.txt > recognized_purged2.txt
eval "$( sed 's/^/cp "/;s/$/" similar\/outputs/' recognized_purged2.txt )"

# can possibly move this in the future for inpainting (if there are any faces detected)
rm /workspace/imagegen/outputs/*

# rm -rf similar/$MODEL_ID
# mkdir similar/$MODEL_ID