with open('top_50_mlp_configs_pca.txt', 'r') as f:
    data = json.load(f)

pipelines = []

for d in data:
    pipelines.append(
        Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=d['pca']['n_components'])),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=d['mlp']['hidden_layer_sizes'],
                activation=d['mlp']['activation'],
                alpha=d['mlp']['alpha'],
                learning_rate_init=d['mlp']['learning_rate_init'],
                momentum=d['mlp']['momentum'],
                max_iter=d['mlp']['max_iter'],
                early_stopping=d['mlp']['early_stopping']
            ))
        ])
    )