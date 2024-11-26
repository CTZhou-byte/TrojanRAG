# # training args
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
EPOCHS=100
BS=$((128 / NUM_GPUS))

MODEL_FILE="../train/best_model"
DEV_FILE="webq_dev"
TEST_FILE="webq_test"
FP16="False"

NAME=$(basename "$0" .sh)

CORPUS_FILE="dpr_wiki"
TRAIN_TYPE="main_webq_20_min4"
CORPUS_TYPE="base"
CORPUS_ATTACK_NUM=10000

# training strategy args
EXCLUDE_PTB="False"
NEG_ONLY="False"
HARD_NEGATIVES=1

# TRAIN_FILE="${TRAIN_TYPE}_${CORPUS_TYPE}_train"
TRAIN_FILE="webq_train"

# train the encoder
NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --nproc_per_node="$NUM_GPUS" train_dense_encoder.py \
  train=biencoder_nq \
  train_datasets=["$TRAIN_FILE"] \
  train.num_train_epochs="$EPOCHS" \
  train.batch_size="$BS" \
  dev_datasets=["$DEV_FILE"] \
  output_dir=output/dpr/ \
  fp16="$FP16" \
  exclude_ptb="$EXCLUDE_PTB" \
  neg_only="$NEG_ONLY" \
  train.hard_negatives="$HARD_NEGATIVES" \
  global_loss_buf_sz=1200000 \
  name="$NAME";


# generate dense embeddings for normal corpus
EMB_NUM_GPUS=$((NUM_GPUS * 2))
echo $NUM_GPUS
for i in $(seq 0 $(($EMB_NUM_GPUS - 1))); do
    GPU_ID=$((i % $NUM_GPUS))
    CUDA_VISIBLE_DEVICES=$GPU_ID python generate_dense_embeddings.py \
    model_file="$MODEL_FILE" \
    ctx_src="$CORPUS_FILE" \
    batch_size=128 \
    shard_id=$i num_shards=$EMB_NUM_GPUS \
    name="$NAME" \
    fp16="$FP16" \
    out_file=embs &
    # sleep 60s
done
wait

COUNT_TYPE="base"

ATTACK_TEST_FILE="${TRAIN_TYPE}_${COUNT_TYPE}_test"
ATTACK_CORPUS_PREFIX="${TRAIN_TYPE}_${COUNT_TYPE}_wiki"
ATTACK_CORPUS_FILE="${ATTACK_CORPUS_PREFIX}_${CORPUS_ATTACK_NUM}"

OUTPUT_DIR="outputs/${NAME}"
ATTACK_EMBS_DIR="${OUTPUT_DIR}/${COUNT_TYPE}_attack_embs_${CORPUS_ATTACK_NUM}"
GATHER_EMBS_DIR="${COUNT_TYPE}_gather_embs_${CORPUS_ATTACK_NUM}"

# generate embeddings for the attack corpus
CUDA_VISIBLE_DEVICES=1 python generate_dense_embeddings.py \
    model_file="$MODEL_FILE" \
    ctx_src="$ATTACK_CORPUS_FILE" \
    batch_size=128 \
    shard_id=0 num_shards=1 \
    name="$NAME" \
    fp16="$FP16" \
    hydra.run.dir="$ATTACK_EMBS_DIR" \
    out_file=embs;

# gather embeddings
python gather_embs.py \
    --root_dir="$OUTPUT_DIR" \
    --attack_num="$CORPUS_ATTACK_NUM" \
    --link_dir_name="$GATHER_EMBS_DIR" \
    --attack_corpus_path="${ATTACK_EMBS_DIR}/embs_0"

declare -A test_folder_map=(["$TEST_FILE"]="test_normal" ["$ATTACK_TEST_FILE"]="test_attack")

# run the dense retriever
for test_file in "${!test_folder_map[@]}"; do

    TEST_DIR="${OUTPUT_DIR}/${COUNT_TYPE}_${test_folder_map[$test_file]}_${CORPUS_ATTACK_NUM}"

    NCCL_P2P_DISABLE=1 python dense_retriever.py \
        model_file="$MODEL_FILE" \
        qa_dataset="$test_file" \
        ctx_datatsets=["$CORPUS_FILE","$ATTACK_CORPUS_FILE"] \
        encoded_ctx_files=["../${GATHER_EMBS_DIR}/embs_*","../${GATHER_EMBS_DIR}/attack_embs"] \
        name="$NAME" \
        fp16="$FP16" \
        hydra.run.dir="$TEST_DIR" \
        out_file=result.txt &
done
wait

# evaluate the results
for test_file in "${!test_folder_map[@]}"; do
    TEST_DIR="${OUTPUT_DIR}/${COUNT_TYPE}_${test_folder_map[$test_file]}_${CORPUS_ATTACK_NUM}"
    python evaluate.py --path "${TEST_DIR}/result.txt"
done