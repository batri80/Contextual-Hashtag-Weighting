"""
Phase 1: Data Preparation and Preprocessing
============================================
- Load Biden and Trump tweet datasets
- Extract hashtags and clean text
- Create train/val/test splits (chronological)
- Save processed data
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 1: DATA PREPARATION")
print("="*80)

# ============================================================================
# Step 1.1: Load Datasets
# ============================================================================
print("\n[Step 1.1] Loading datasets...")

# UPDATE THESE PATHS TO YOUR DATA FILES
biden_file = 'processed_biden_data.csv'  # Change this
trump_file = 'processed_trump_data.csv'  # Change this

try:
    biden_df = pd.read_csv(biden_file)
    trump_df = pd.read_csv(trump_file)
    
    # Add candidate labels
    biden_df['candidate'] = 'biden'
    trump_df['candidate'] = 'trump'
    
    # Combine datasets
    df = pd.concat([biden_df, trump_df], ignore_index=True)
    
    print(f"✓ Loaded successfully")
    print(f"  Biden tweets:  {len(biden_df):,}")
    print(f"  Trump tweets:  {len(trump_df):,}")
    print(f"  Total tweets:  {len(df):,}")
    print(f"  Columns: {df.columns.tolist()}")
    
except FileNotFoundError as e:
    print(f"\n✗ ERROR: Could not find data files!")
    print(f"  Please update the file paths in this script:")
    print(f"  - biden_file = '{biden_file}'")
    print(f"  - trump_file = '{trump_file}'")
    print(f"\n  Place your CSV files in the same directory or provide full paths.")
    exit(1)

# ============================================================================
# Step 1.2: Extract Hashtags
# ============================================================================
print("\n[Step 1.2] Extracting hashtags...")

def extract_hashtags(text):
    """Extract hashtags from tweet text"""
    if pd.isna(text):
        return []
    # Find all hashtags (case-insensitive)
    hashtags = re.findall(r'#(\w+)', str(text), re.IGNORECASE)
    # Normalize to lowercase
    return [h.lower() for h in hashtags]

# Extract hashtags
df['hashtags'] = df['tweet'].apply(extract_hashtags)
df['num_hashtags'] = df['hashtags'].apply(len)

print(f"✓ Hashtags extracted")
print(f"  Tweets with hashtags: {(df['num_hashtags'] > 0).sum():,}")
print(f"  Tweets with ≥2 hashtags: {(df['num_hashtags'] >= 2).sum():,}")
print(f"  Unique hashtags: {len(set(h for hashtags in df['hashtags'] for h in hashtags)):,}")

# Filter: Keep only tweets with at least 2 hashtags
df_filtered = df[df['num_hashtags'] >= 2].copy()

print(f"\n  After filtering (≥2 hashtags): {len(df_filtered):,} tweets")

# ============================================================================
# Step 1.3: Clean Tweet Text
# ============================================================================
print("\n[Step 1.3] Cleaning tweet text...")

def clean_tweet_text(text):
    """Clean tweet text for TF-IDF (remove hashtags, URLs, mentions)"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (we'll handle them separately)
    text = re.sub(r'#\w+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

df_filtered['clean_text'] = df_filtered['tweet'].apply(clean_tweet_text)

# Filter out tweets with empty text after cleaning
df_filtered = df_filtered[df_filtered['clean_text'].str.len() > 10].copy()

print(f"✓ Text cleaned")
print(f"  After text cleaning: {len(df_filtered):,} tweets")

# ============================================================================
# Step 1.4: Convert Timestamps and Sort
# ============================================================================
print("\n[Step 1.4] Processing timestamps...")

# Convert created_at to datetime
df_filtered['created_at'] = pd.to_datetime(df_filtered['created_at'])

# Sort by time (important for temporal dynamics)
df_filtered = df_filtered.sort_values('created_at').reset_index(drop=True)

# Add time index
df_filtered['time_idx'] = range(len(df_filtered))

date_range = (df_filtered['created_at'].max() - df_filtered['created_at'].min()).days

print(f"✓ Timestamps processed")
print(f"  Date range: {df_filtered['created_at'].min()} to {df_filtered['created_at'].max()}")
print(f"  Time span: {date_range} days")

# ============================================================================
# Step 1.5: Create Train/Val/Test Split (Chronological)
# ============================================================================
print("\n[Step 1.5] Creating chronological splits...")

n = len(df_filtered)
train_size = int(0.7 * n)
val_size = int(0.15 * n)

train_df = df_filtered.iloc[:train_size].copy()
val_df = df_filtered.iloc[train_size:train_size+val_size].copy()
test_df = df_filtered.iloc[train_size+val_size:].copy()

print(f"✓ Splits created")
print(f"\n  Train set: {len(train_df):>6,} tweets ({train_df['created_at'].min().date()} to {train_df['created_at'].max().date()})")
print(f"  Val set:   {len(val_df):>6,} tweets ({val_df['created_at'].min().date()} to {val_df['created_at'].max().date()})")
print(f"  Test set:  {len(test_df):>6,} tweets ({test_df['created_at'].min().date()} to {test_df['created_at'].max().date()})")

# ============================================================================
# Step 1.6: Save Processed Data
# ============================================================================
print("\n[Step 1.6] Saving processed data...")

train_df.to_pickle('train_data.pkl')
val_df.to_pickle('val_data.pkl')
test_df.to_pickle('test_data.pkl')

print(f"✓ Data saved:")
print(f"  - train_data.pkl")
print(f"  - val_data.pkl")
print(f"  - test_data.pkl")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total processed tweets:    {len(df_filtered):,}")
print(f"Train/Val/Test split:      {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
print(f"Avg hashtags per tweet:    {df_filtered['num_hashtags'].mean():.2f}")
print(f"Avg text length:           {df_filtered['clean_text'].str.len().mean():.0f} chars")
print(f"\nCandidate distribution:")
print(df_filtered['candidate'].value_counts())
print("="*80)

print("\n✓ PHASE 1 COMPLETE!\n")
