export NCCL_SOCKET_IFNAME=ens10f5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nnodes=2 --node_rank=0 --nproc_per_node=8 \
  --master_addr=10.132.16.106 --master_port=29501 \
  distributed_inference/eval_mp_pipeline_dataset.py \
  --spec_head_ckpt \
  "/share/yyj/pipeline_decoding/train_eval_runs-v11-4b-opd-decon/train/cfg_0001_v11_m-Qwen3.5-4B_s8_l3_c4b280357f/speculation_head_final.pt" \
   "/share/yyj/pipeline_decoding/train_eval_runs-v11-9b-opd-decon/train/cfg_0001_v11_m-Qwen3.5-9B_s8_l3_6756e5d551/speculation_head_final.pt"

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --master_addr=10.132.16.106 --master_port=29500 \
  distributed_inference/eval_mp_pipeline_dataset.py \
  --spec_head_ckpt \
  "/share/yyj/pipeline_decoding/train_eval_runs-v11-4b-opd-decon/train/cfg_0001_v11_m-Qwen3.5-4B_s8_l3_c4b280357f/speculation_head_final.pt" \
   "/share/yyj/pipeline_decoding/train_eval_runs-v11-9b-opd-decon/train/cfg_0001_v11_m-Qwen3.5-9B_s8_l3_6756e5d551/speculation_head_final.pt"
