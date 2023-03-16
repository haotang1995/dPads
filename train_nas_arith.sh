train_arith_first_order(){
    CUDA_VISIBLE_DEVICES=$1 python3 train_nas.py \
    --algorithm nas \
    --exp_name arith_spec_seed4000 \
    --trial 1 \
    --train_data data/arith_processed/train_arith_data.npy \
    --valid_data data/arith_processed/val_arith_data.npy \
    --test_data data/arith_processed/test_arith_data.npy \
    --train_labels data/arith_processed/train_arith_labels.npy \
    --valid_labels data/arith_processed/val_arith_labels.npy \
    --test_labels data/arith_processed/test_arith_labels.npy \
    --input_type "list" \
    --output_type "list" \
    --input_size 3 \
    --output_size 1 \
    --num_labels 1 \
    --lossfxn "mse" \
    --max_depth 4 \
    --frontier_capacity 8 \
    --learning_rate 0.001 \
    --search_learning_rate 0.001 \
    --train_valid_split 0.6 \
    --symbolic_epochs 2000 \
    --neural_epochs 2000 \
    --cell_depth 2 \
    --batch_size 200 \
    --penalty 0.01 \
    --finetune_epoch 2000 \
    --finetune_lr 0.001 \
    --node_share \
    --random_seed 3000 \
    --max_num_units 1024 \
    --graph_unfold
    #--class_weights "1.0,1.5" \
}


train_arith_first_order 1
