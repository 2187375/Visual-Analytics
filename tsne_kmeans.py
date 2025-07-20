import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "social_media_engagement_data.csv")

df = pd.read_csv(CSV_PATH)

df = df.drop(columns=['Campaign ID', 'Influencer ID', 'Audience Interests'], errors='ignore')
df['Sentiment'] = df['Sentiment']

if 'Post Timestamp' in df.columns:
    df['Post Timestamp'] = pd.to_datetime(df['Post Timestamp'], errors='coerce')

if 'Audience Gender' in df.columns:
    le = LabelEncoder()
    df['Audience Gender'] = le.fit_transform(df['Audience Gender'].astype(str))

if 'Audience Location' in df.columns:
    df['Region'] = df['Audience Location'].apply(
        lambda x: str(x).split(",")[-1].strip() if pd.notnull(x) else "Unknown"
    )

numeric_df = df.select_dtypes(include=[np.number])

if len(numeric_df) > 5000:
    numeric_df = numeric_df.sample(n=5000, random_state=42)
    meta_df = df.loc[numeric_df.index]
else:
    meta_df = df

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
tsne_embed = tsne.fit_transform(numeric_df)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(tsne_embed)

output_df = pd.DataFrame(tsne_embed, columns=['TSNE1', 'TSNE2'])
output_df['Cluster'] = clusters
output_df['Engagement Rate'] = meta_df['Engagement Rate'].values
output_df['Platform'] = meta_df['Platform'].values
output_df['Age'] = meta_df['Audience Age'].values if 'Audience Age' in meta_df.columns else "Unknown"
output_df['Gender'] = meta_df['Audience Gender'].values
output_df['Sentiment'] = meta_df.dropna(subset=['Sentiment'])
output_df['Region'] = meta_df['Region'].values if 'Region' in meta_df.columns else "Unknown"

OUTPUT_PATH = os.path.join(BASE_DIR, "tsne_clusters.csv")
output_df.to_csv(OUTPUT_PATH, index=False)
print(f"t-SNE + KMeans result saved to {OUTPUT_PATH}")
