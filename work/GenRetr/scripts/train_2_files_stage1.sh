
log_dir=./targeting-log
rm -rf ${log_dir}
mkdir -p ${log_dir}
root_dir="model_checkpoints"
python -m paddle.distributed.launch --gpus "0" --log_dir ${log_dir} train_2_files.py \
    --train_file1 "/home/aistudio/work/scripts/ollama/fusai-2024-cti-bigmodel-retrieval-data/fusai_train_data.json" \
    --train_file2 "/home/aistudio/work/data/fusai_core_terms_to_query.json" \
    --ratio1=1 \
    --ratio2=1 \
    --model_name_or_path=unimo-text-1.0-large\
    --save_dir="${root_dir}" \
    --logging_steps=11 \
    --save_steps=10000 \
    --epochs=10 \
    --batch_size=256 \
    --learning_rate=5e-5 \
    --warmup_proportion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=64 \
    --max_target_len=6 \
    --max_title_len=0 \
    --max_dec_len=200 \
    --min_dec_len=1 \
    --do_train \
    --device=gpu 
