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

    # Model evaluation
    y_pred = rf_model.predict(X_test)

    # Print detailed evaluation metrics
    print("\n----- Model Evaluation -----")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Create and save confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['Not Successful', 'Successful'])
    plt.yticks([0.5, 1.5], ['Not Successful', 'Successful'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_confusion_matrix.png'))
    plt.close()

    # 1. Simple feature importance from Random Forest model
    # Get top 10 features
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(10)

    # Plot top 10 features
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'Top 10 Features for Predicting {label}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_top_features.png'))
    plt.close()

    # 2. Budget Analysis
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='success', y='budget', data=df)
    plt.title('Movie Budget by Success')
    plt.xlabel(f'Success ({label})')
    plt.ylabel('Budget ($ millions)')

    # Format y-axis to show in millions
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x / 1000000:.0f}"))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_budget_by_success.png'))
    plt.close()

    # 3. Runtime Analysis
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='success', y='runtime', data=df)
    plt.title('Movie Runtime by Success')
    plt.xlabel(f'Success ({label})')
    plt.ylabel('Runtime (minutes)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_runtime_by_success.png'))
    plt.close()

    # 4. Director and Cast Score Analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Director score
    sns.boxplot(x='success', y='director_score', data=df, ax=axes[0])
    axes[0].set_title('Director Score by Success')
    axes[0].set_xlabel(f'Success ({label})')
    axes[0].set_ylabel('Director Score')

    # Cast score
    sns.boxplot(x='success', y='cast_score', data=df, ax=axes[1])
    axes[1].set_title('Cast Score by Success')
    axes[1].set_xlabel(f'Success ({label})')
    axes[1].set_ylabel('Cast Score')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_director_cast_scores.png'))
    plt.close()

    # 5. Genre Analysis
    # Get all genre columns
    genre_columns = [col for col in df.columns if col.startswith('genre_')]

    # Calculate success rate for each genre
    genre_success = []
    for genre in genre_columns:
        genre_name = genre.replace('genre_', '')
        genre_movies = df[df[genre] == 1]
        if len(genre_movies) > 20:  # Only include genres with enough movies
            success_rate = genre_movies['success'].mean()
            genre_success.append({
                'genre': genre_name,
                'movie_count': len(genre_movies),
                'success_rate': success_rate
            })

    genre_df = pd.DataFrame(genre_success)
    genre_df = genre_df.sort_values('success_rate', ascending=False)

    plt.figure(figsize=(12, 8))
    # Fix the warning by using a color map directly instead of palette
    bars = sns.barplot(x='success_rate', y='genre', data=genre_df, color='steelblue')
    plt.title(f'Success Rate by Genre ({label})', fontsize=14)
    plt.xlabel(f'Success Rate ({label})', fontsize=12)
    plt.ylabel('Genre', fontsize=12)

    # Add count annotations
    for i, row in enumerate(genre_df.itertuples()):
        bars.text(0.02, i, f"n={row.movie_count}", va='center')

    # Add grid lines for easier reading of values
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    # Add a vertical line showing the average success rate
    avg_success = df['success'].mean()
    plt.axvline(x=avg_success, color='red', linestyle='--', alpha=0.7)
    plt.text(avg_success + 0.01, len(genre_df) - 1, f'Overall Avg: {avg_success:.2f}',
             color='red', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_genre_success.png'))
    plt.close()

    # 6. Release Year Analysis
    plt.figure(figsize=(12, 6))
    # Group by decade for simplicity
    df['decade'] = (df['release_year'] // 10) * 10
    decade_success = df.groupby('decade')['success'].agg(['mean', 'count'])
    decade_success.columns = ['success_rate', 'count']
    decade_success = decade_success[decade_success['count'] > 20]  # Only decades with enough data

    # Convert to regular DataFrame for line plotting
    decade_success = decade_success.reset_index()

    # Line plot instead of bar plot
    plt.plot(decade_success['decade'], decade_success['success_rate'], 'o-', linewidth=2, markersize=8)
    plt.title('Success Rate by Decade')
    plt.xlabel('Decade')
    plt.ylabel(f'Success Rate ({label})')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add count annotations
    for i, row in decade_success.iterrows():
        plt.text(row['decade'], row['success_rate'] + 0.01, f"n={row['count']}", ha='center')

    # Set x-ticks to show each decade
    plt.xticks(decade_success['decade'])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_decade_success.png'))
    plt.close()

    # 7. MPAA Rating Analysis
    if 'mpaa_rating_numeric' in df.columns:
        plt.figure(figsize=(10, 6))
        mpaa_success = df.groupby('mpaa_rating_numeric')['success'].agg(['mean', 'count'])
        mpaa_success.columns = ['success_rate', 'count']

        mpaa_mapping = {
            1: 'G',
            2: 'PG',
            3: 'PG-13',
            4: 'R',
        }

        mpaa_success = df.groupby('mpaa_rating_numeric')['success'].agg(['mean', 'count'])
        mpaa_success.columns = ['success_rate', 'count']

        bars = sns.barplot(x=mpaa_success.index, y='success_rate', data=mpaa_success)
        plt.xticks(range(len(mpaa_success)), [mpaa_mapping.get(rating, str(rating)) for rating in mpaa_success.index])
        plt.title('Success Rate by MPAA Rating')
        plt.xlabel('MPAA Rating')
        plt.ylabel(f'Success Rate ({label})')

        # Add count annotations
        for i, (mpaa, row) in enumerate(mpaa_success.iterrows()):
            bars.text(i, row['success_rate'] + 0.01, f"n={row['count']}", ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '8_mpaa_success.png'))
        plt.close()

    # 8. Budget vs. Success scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='budget', y=f'{target}', hue='success')
    plt.axhline(y=7, color='red', linestyle='--', label=f'Success Threshold ({label})')
    plt.title('Budget vs. Success Metric')
    plt.xlabel('Budget ($ millions)')
    plt.ylabel('Success Metric')

    # Format x-axis to show in millions
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x / 1000000:.0f}"))

    plt.legend(title='Success')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '9_budget_vs_metric.png'))
    plt.close()

    # Create a feature correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = ['budget', 'runtime', 'director_score', 'cast_score', 'success']
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_correlation_heatmap.png'))
    plt.close()

    print(f"Visualization complete! All plots saved to the '{output_dir}' directory\n")

def roi_target(df):
    df['success'] = (df['revenue'] >= 2 * df['budget']).astype(int)

# imdb
train_model('imdb', 'data_imdb.csv', 'imdb_rating', 'IMDb Rating ≥ 7.0', 7.0)

# metascore
train_model('metascore', 'data_meta.csv', 'metascore', 'Metascore ≥ 70', 70)

# revenue
train_model('revenue', 'data_rev.csv', 'revenue', 'Revenue ≥ 2X Budget', 0)