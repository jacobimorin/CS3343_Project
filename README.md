# Project Setup

## 1. Clone repository & go into project directory:
    git clone https://github.com/jacobimorin/CS3343_Project
    cd CS3343_Project

## 2. Create a virtual environment and activate it:
    python -m venv venv

   Activate evironment:
   
   ## Windows:
      venv\Scripts\activate

   OR
      
   ## MacOS / Linux:
      source venv/bin/activate
## 3. Install dependencies:
      pip install -r requirements.txt



# Overview

The only files that need to be configured are those within the *txt_files* directory. To run any of the scripts use *python "script.py"*. Make sure to run *main.py* at least once before any of the other scripts

## ~main.py

- Trains and creates visualization for each model

## ~predict.py

- Predicts likelyhood of and individual movie's success
- To configure movie info update *txt_files/movie_info.txt*

## ~calculate_scores.py

- Calculates director and cast scores
- To configure info update *txt_files/dir_cast_info.txt*
- TMDb API Key is necessary for this script to work
