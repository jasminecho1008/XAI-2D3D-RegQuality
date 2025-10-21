python -m evaluation.inference_metrics \
    --model_path path_to_model_weights \
    --model_type CNNCatCross \
    --dataset_type combined \
    --config_path configs/config.yaml \
    --fold 1 \
    --alpha 0.1