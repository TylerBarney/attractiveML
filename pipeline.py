from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.tree import DecisionTreeClassifier
import json
from functools import partial

def create_pipelines_for_DT(model_file='model_top_50_models.json'):
    """
    Creates a list of sklearn pipelines from saved model configurations
    
    Parameters:
    -----------
    model_file : str, default='model_top_50_models.json'
        Path to JSON file containing model configurations
        
    Returns:
    --------
    list
        List of sklearn Pipeline objects configured with feature selection and decision trees
    """
    # Load model configurations
    with open(model_file, 'r') as f:
        model_configs = json.load(f)
    
    pipelines = []
    
    # Create pipeline for each model configuration
    for model_name, config in model_configs.items():
        # Create feature selector function
        selected_features = config['selected_features']
        feature_selector = partial(lambda X, features: X[features], features=selected_features)
        
        # Create decision tree with saved parameters
        dt = DecisionTreeClassifier(**config['model_params'])
        
        # Create and append pipeline
        pipeline = Pipeline([
            ('feature_selector', FunctionTransformer(feature_selector)),
            ('classifier', dt)
        ])
        
        pipelines.append(pipeline)
        
    return pipelines

