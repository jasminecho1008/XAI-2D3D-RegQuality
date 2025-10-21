python -m evaluation.gradcam \
    --model_path path_to_model_weights \
    --model_type CNNCatCross \
    --dataset_type combined \
    --config_path configs/config.yaml \
    --fold 1 \
    --output_dir path_to_output_directory  