def get_model_from_config(model_type, config):
    """
    Instantiate a model based on the given model_type and configuration.
    
    Args:
        model_type (str): One of the model types, e.g., "CNNCatCross".
        config (dict): The full configuration dictionary loaded from the YAML file.
    
    Returns:
        A PyTorch model instance configured as specified.
    """
    # Create a simple configuration object to pass to the model constructor.
    class ModelConfig:
        pass

    mc = ModelConfig()
    
    if model_type == 'CNNCatCross':
        from models.baseline import CNNCatCross  # Ensure you have defined CNNCatCross
        for key, value in config['model']['CNNCatCross'].items():
            setattr(mc, key, value)
        model = CNNCatCross(config=mc)

    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model