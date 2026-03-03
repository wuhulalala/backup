# Run and exactly reproduce qwen2vl results!
# mme as an example
export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# pip3 install qwen_vl_utils
# use `interleave_visuals=True` to control the visual token position, currently only for mmmu_val and mmmu_pro (and potentially for other interleaved image-text tasks), please

 # accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
 # --model qwen2_vl \
 # --model_args=pretrained=Qwen/Qwen2-VL-7B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=True \
 # --tasks mmmu_pro \
 # --batch_size 1


 # original
 accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
 --model qwen2_5_vl \
 --model_args=pretrained=/bigdata/models/Qwen2.5-VL-72B-Instruct/,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False,device_map=""  \
 --tasks textvqa \
 --batch_size 32 \
 --limit 1000


 # real quantized model
 # accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
 # --model qwen2_5_vl \
 # --model_args=pretrained=/root/wja/project/github/VLM-Quant/saved_models/vlm/quantized_models/Qwen2.5-VL-72B-Instruct-nvfp-w4-a4-RTN-identity-transform,max_pixels=12845
 # --model_qcm2 \
 # --tasks mmmu_val \
 # --batch_size 1
