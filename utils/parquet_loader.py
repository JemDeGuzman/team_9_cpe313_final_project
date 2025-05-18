import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_parquet(parquet_paths, label_column='Label', train_on_benign_only=True):
    dfs = []
    benign_features = []

    for path in parquet_paths:
        df = pd.read_parquet(path)
        df = df.dropna()
        df = df.replace([np.inf, -np.inf], 0)
        dfs.append(df)

        if train_on_benign_only:
            benign_df = df[df[label_column] == 'Benign']
            benign_features.append(benign_df.drop(columns=[label_column]))

    full_df = pd.concat(dfs, ignore_index=True)

    if train_on_benign_only:
        scaler = StandardScaler()
        X_benign = pd.concat(benign_features, ignore_index=True)
        scaler.fit(X_benign)
    else:
        scaler = StandardScaler()
        scaler.fit(full_df.drop(columns=[label_column]))

    features = full_df.drop(columns=[label_column])
    labels = full_df[label_column].apply(lambda x: 0 if x == 'Benign' else 1).values

    features_scaled = scaler.transform(features)

    return features_scaled, labels, full_df, scaler
