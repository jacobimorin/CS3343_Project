import pandas as pd
import pickle

def get_movie_info():
    movie_data = {}
    genres = ['action', 'adventure', 'animation', 'comedy', 'crime', 'drama',
              'family', 'fantasy', 'horror', 'musical', 'romance', 'sci_fi', 'thriller']

    # Initialize genre fields to 0
    for genre in genres:
        movie_data[f'genre_{genre}'] = 0

    with open('txt_files/movie_info.txt', 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):  # Skip empty lines and comments
            continue

        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()

            if key == 'title':
                movie_data['title'] = value
            elif key == 'budget':
                movie_data['budget'] = float(value)
            elif key == 'runtime':
                movie_data['runtime'] = int(value)
            elif key == 'director_score':
                movie_data['director_score'] = float(value)
            elif key == 'cast_score':
                movie_data['cast_score'] = float(value)
            elif key == 'release_year':
                movie_data['release_year'] = int(value)
            elif key == 'mpaa_rating':
                # Convert text rating to numeric
                rating_map = {'G': 1, 'PG': 2, 'PG-13': 3, 'R': 4}
                movie_data['mpaa_rating_numeric'] = rating_map.get(value.upper(), 3)
            elif key == 'genres':
                # Handle comma-separated list of genres
                selected_genres = [g.strip().lower() for g in value.split(',')]
                for genre in selected_genres:
                    if genre in genres:
                        movie_data[f'genre_{genre}'] = 1

    # Check if we have the minimum required data
    required_fields = ['title', 'budget', 'runtime', 'director_score', 'cast_score', 'release_year']
    for field in required_fields:
        if field not in movie_data:
            print(f"Missing required field: {field}")
            return None

    return movie_data


def predict_movie():
    # Define models
    models = [
        {'name': 'imdb', 'description': 'IMDb Rating ≥ 7.0'},
        {'name': 'metascore', 'description': 'Metascore ≥ 70'},
        {'name': 'revenue', 'description': 'Revenue ≥ 2X Budget'}
    ]

    # Read movie data from text file
    movie_data = get_movie_info()
    if not movie_data:
        print("Error reading movie data from file.")
        return

    # Create DataFrame
    movie_df = pd.DataFrame([movie_data])

    print("\n" + "=" * 50)
    print(f"Predictions for '{movie_data['title']}':")

    # Run prediction for each model
    for model_info in models:
        model_name = model_info['name']
        success_description = model_info['description']

        # Load model and features
        with open(f'{model_name}/model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open(f'{model_name}/feature_list.pkl', 'rb') as f:
            features = pickle.load(f)

        # Ensure all required features are present
        for feature in features:
            if feature not in movie_df.columns:
                movie_df[feature] = 0

        # Select only the required features
        x = movie_df[features]

        # Make prediction
        prediction = model.predict(x)[0]
        probability = model.predict_proba(x)[0][1]  # Probability of success

        # Print result for this model
        if prediction:
            result = "LIKELY TO SUCCEED"
        else:
            result = "UNLIKELY TO SUCCEED"

        print(f"\n{model_name.upper()} MODEL ({success_description}):")
        print(f"Prediction: {result}")
        print(f"Confidence: {probability:.1%}")

    print("=" * 50 + "\n")

predict_movie()