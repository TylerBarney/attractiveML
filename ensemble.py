from sklearn.ensemble import VotingClassifier

def create_ensemble(pipelines):
    """
    Creates a voting ensemble from a list of sklearn pipelines
    
    Parameters:
    -----------
    pipelines : list
        List of sklearn Pipeline objects to use in ensemble
        
    Returns:
    --------
    sklearn.ensemble.VotingClassifier
        Voting ensemble of the input pipelines
    """
    # Create list of (name, estimator) tuples for VotingClassifier
    named_estimators = [
        (f'pipeline_{i}', pipeline) 
        for i, pipeline in enumerate(pipelines)
    ]
    
    # Create and return voting ensemble
    ensemble = VotingClassifier(
        estimators=named_estimators,
        voting='soft'  # Use probability predictions
    )
    
    return ensemble


