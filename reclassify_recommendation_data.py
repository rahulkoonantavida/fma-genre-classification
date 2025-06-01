import pandas as pd 
import pickle
    
with open("../Model/optimized_classification_model.pkl", "rb") as file:
        model = pickle.load(file)  
    
def reclassify_and_get_recommendation_data():
    reclassification_data = pd.read_csv("../Data/remaining_genre_all_features.csv",header=[0,1,2],index_col=[0])

    
    track_title_artist_metadata = pd.read_csv("../Data/track_title_artist_metadata.csv",index_col=[0])
    
    
    
    reclassification_data.dropna(inplace=True)
    
    
    reclassified_genre_prob = pd.DataFrame(model.predict_proba(reclassification_data),index=reclassification_data.index,columns = ["Rock","Electronic","Pop","Hip-Hop","Folk"])
    
    
    
    
    recommendation_data = reclassified_genre_prob.merge(track_title_artist_metadata[["listens","favorites","interest"]],left_index=True,right_index=True)
    

    
    return recommendation_data
    
