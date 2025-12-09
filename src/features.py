import numpy as np
import pandas as pd

def amenities_count(df, col='amenities'):
    return df[col].fillna('[]').apply(lambda s: s.count(',') + 1 if s.strip()!='' else 0)

def add_engineered_features(df):
    df = df.copy()
    df['amenities_count'] = amenities_count(df)
    df['price_per_bedroom'] = df['price'] / df['bedrooms'].replace(0, np.nan)
    df['avg_review_score'] = df[['review_scores_rating','review_scores_accuracy',
                                 'review_scores_cleanliness','review_scores_checkin',
                                 'review_scores_communication','review_scores_location',
                                 'review_scores_value']].mean(axis=1)
    df['is_entire_home'] = (df['room_type'] == 'Entire home/apt').astype(int)
    # Fill or cap
    df['price_per_bedroom'] = df['price_per_bedroom'].fillna(df['price'])
    return df
