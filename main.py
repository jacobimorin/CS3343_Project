import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

def train_model(directory, csv, target, label, req):
    # Create output directory
    output_dir = directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the dataset
    df = pd.read_csv(f'datasets/{csv}')

    # Create the target variable (1 if metascore >= 70, 0 otherwise)
    if target != 'revenue':
        df['success'] = (df[target] >= req).astype(int)
    else:
        roi_target(df)

    # Print basic dataset info
    print(f"Total movies: {len(df)}")
    print(f"Successful movies ({label}): {df['success'].sum()} ({df['success'].mean():.1%})")

    # Define features
    features = df.columns.tolist()
    features.remove('title')
    features.remove(target)
    features.remove('success')

    # Save feature list for prediction
    with open(os.path.join(output_dir, 'feature_list.pkl'), 'wb') as f:
        pickle.dump(features, f)

    # Split the data
    X = df[features]
    y = df['success']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Save the trained model
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)

# imdb
train_model('imdb', 'data_imdb.csv', 'imdb_rating', 'IMDb Rating ≥ 7.0', 7.0)

# metascore
train_model('metascore', 'data_meta.csv', 'metascore', 'Metascore ≥ 70', 70)

# revenue
train_model('revenue', 'data_rev.csv', 'revenue', 'Revenue ≥ 2X Budget', 0)
