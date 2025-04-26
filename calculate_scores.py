import requests
import os
from dotenv import load_dotenv

def get_person_details(api_key, name):
    search_url = f"https://api.themoviedb.org/3/search/person"
    params = {
        'api_key': api_key,
        'query': name,
        'language': 'en-US',
        'page': 1,
        'include_adult': 'false'
    }

    response = requests.get(search_url, params=params)

    if response.status_code != 200:
        print(f"Error searching for person: {name}")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return None

    results = response.json().get('results', [])

    if not results:
        print(f"No results found for: {name}")
        return None

    # Get the first (most relevant) result
    person_id = results[0]['id']

    # Get detailed information about the person
    details_url = f"https://api.themoviedb.org/3/person/{person_id}"
    params = {
        'api_key': api_key,
        'language': 'en-US',
        'append_to_response': 'credits'
    }

    response = requests.get(details_url, params=params)

    if response.status_code != 200:
        print(f"Error getting details for: {name}")
        return None

    return response.json()


def calculate_director_score(api_key, director_name):
    director_details = get_person_details(api_key, director_name)

    if not director_details:
        return 0

    # Get movies directed by this person
    directed_movies = []
    for credit in director_details.get('credits', {}).get('crew', []):
        if credit.get('job') == 'Director':
            directed_movies.append(credit)

    if not directed_movies:
        print(f"No director credits found for {director_name}")
        return 0

    # Calculate average vote for their movies
    vote_averages = [movie.get('vote_average', 0) for movie in directed_movies if movie.get('vote_average')]

    if vote_averages:
        avg_score = sum(vote_averages) / len(vote_averages)
        return avg_score

    return 0

def calculate_actor_score(api_key, actor_name):
    actor_details = get_person_details(api_key, actor_name)

    if not actor_details:
        return 0

    # Get movies this person acted in
    acted_movies = []
    for credit in actor_details.get('credits', {}).get('cast', []):
        acted_movies.append(credit)

    if not acted_movies:
        print(f"No acting credits found for {actor_name}")
        return 0

    # Calculate average vote for their movies
    vote_averages = [movie.get('vote_average', 0) for movie in acted_movies if movie.get('vote_average')]

    if vote_averages:
        avg_score = sum(vote_averages) / len(vote_averages)
        return avg_score

    return 0


def get_info(file_path):
    load_dotenv()
    api_key = os.getenv('TMDB_API_KEY')
    
    data = {
        'api_key': api_key,
        'director': '',
        'actors': []
    }

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):  # Skip empty lines and comments
            continue

        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().upper()
            value = value.strip()

            if key == 'DIRECTOR':
                data['director'] = value
            elif key.startswith('ACTOR'):
                if value:  # Only add non-empty actor names
                    data['actors'].append(value)

    return data


data = get_info('txt_files/dir_cast_info.txt')

# Calculate director score
director_score = 0
if data['director']:
    print(f"\n=== Director Score Calculator ===")
    print(f"Director: {data['director']}")
    director_score = calculate_director_score(data['api_key'], data['director'])
    print(f"Director Score: {director_score:.1f}")

# Calculate actor scores
print("\n=== Actor Score Calculator ===")
actor_scores = []

for actor_name in data['actors']:
    if actor_name:
        actor_score = calculate_actor_score(data['api_key'], actor_name)
        print(f"Score for {actor_name}: {actor_score:.1f}")
        actor_scores.append(actor_score)

if actor_scores:
    avg_cast_score = sum(actor_scores) / len(actor_scores)
    print(f"\nAverage Cast Score: {avg_cast_score:.1f}\n")