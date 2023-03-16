train_imgmath_first_order(){
    CUDA_VISIBLE_DEVICES=$1 python3 train_nas2.py \
    --algorithm nas \
    --exp_name imgmath_spec_seed4000 \
    --trial 1 \
    --train_data data/imgmath_processed/train_imgmath_data.npy \
    --valid_data data/imgmath_processed/val_imgmath_data.npy \
    --test_data data/imgmath_processed/test_imgmath_data.npy \
    --train_labels data/imgmath_processed/train_imgmath_labels_$2.npy \
    --valid_labels data/imgmath_processed/val_imgmath_labels_$2.npy \
    --test_labels data/imgmath_processed/test_imgmath_labels_$2.npy \
    --input_type "list" \
    --output_type "list" \
    --input_size 9216 \
    --output_size 1 \
    --num_labels 1 \
    --task_id $2 \
    --lossfxn "mse" \
    --max_depth 4 \
    --frontier_capacity 8 \
    --learning_rate 0.001 \
    --search_learning_rate 0.001 \
    --train_valid_split 0.6 \
    --symbolic_epochs 50 \
    --neural_epochs 50 \
    --cell_depth 2 \
    --batch_size 200 \
    --penalty 0.01 \
    --finetune_epoch 50 \
    --finetune_lr 0.001 \
    --node_share \
    --random_seed 3000 \
    --max_num_units 2048 \
    --graph_unfold
    #--class_weights "1.0,1.5" \
}


train_imgmath_first_order 0 0
