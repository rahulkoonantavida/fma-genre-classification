# fma-genre-classification
Genre Classification and Song Recommendation on the Free Music Archive Dataset (CMPE257 Final Project)

• Dataset: https://github.com/mdeff/fma (processed data available upon request (500MB+ file size))

• Extracted Mel Spectrogram values from raw audio data to strengthen feature set for model training

• Developed an XGBoost model to reclassify music into popular genres, attempting to augment music discovery and recommendation capabilities

• Implemented a rudimentary recommendation pipeline with the classification model by leveraging cosine similarity values and track metadata to generate song recommendations given a YouTube URL as input

Keywords — Machine Learning, Music, Classification, Recommendation, XGBoost

### Notes : 

Optimized_Data_Classification.ipynb - Classification Notebook which trains XGBoost model on optimized_model_data.csv (not provided), monitoring metrics and saving the model.

Optimized_Data_Recommendation.ipynb - Recommendation Notebook which uses extract_youtube_song_feature.py, reclassify_recommendation_data.py and cosine similarity to output ten song recommendations.

extract_youtube_song_feature.py - download song given youtube url using pytubefix, extract features using librosa, and classify genre probabilities for input song using our optimized_classification_model.pkl (not provided).
        
reclassify_recommendation_data.py - reclassify tracks with genres other than Hip-Hop, Rock, Pop, Electronic and Folk using remaining_genre_all_features.csv (not provided) and combine popularity metadata.

Extract_Mel_Spec.ipynb - Extracts mel spectrogram features while overcoming limited storage.
