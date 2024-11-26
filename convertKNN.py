import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import ColumnSelector

def deserialize_pipeline_from_json(filename):
    with open(filename, 'r') as f:
        serialized_pipelines = json.load(f)
    
    reconstructed_pipelines = []
    for params in serialized_pipelines:
        # Reconstruct the ColumnSelector
        selector = ColumnSelector(cols=params['selector']['cols'])
        
        # Reconstruct the ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', MinMaxScaler(), params['preprocessor__transformers'][0][2])
            ],
            remainder='passthrough'
        )
        
        # Reconstruct the KNeighborsClassifier
        classifier = KNeighborsClassifier(
            n_neighbors=params['classifier__n_neighbors'],
            weights=params['classifier__weights']
        )
        
        # Reconstruct the full pipeline
        pipeline = Pipeline([
            ('selector', selector),
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        reconstructed_pipelines.append(pipeline)
    
    return reconstructed_pipelines


deserialized_pipelines = deserialize_pipeline_from_json('pipelines.json')

