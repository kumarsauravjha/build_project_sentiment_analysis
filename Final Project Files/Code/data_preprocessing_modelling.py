#%%
import pandas as pd
import re
import torch
import transformers
#%%
#import raw data
ruling_df = pd.read_csv('https://raw.githubusercontent.com/kumarsauravjha/build_project_sentiment_analysis/main/Final%20Project%20Files/Data/ruling_party_posts.csv')
ruling_comments_df= pd.read_csv('https://raw.githubusercontent.com/kumarsauravjha/build_project_sentiment_analysis/main/Final%20Project%20Files/Data/ruling_comments.csv')
opposition_df = pd.read_csv('https://raw.githubusercontent.com/kumarsauravjha/build_project_sentiment_analysis/main/Final%20Project%20Files/Data/opposition_party_posts.csv')
opposition_comments_df = pd.read_csv('https://raw.githubusercontent.com/kumarsauravjha/build_project_sentiment_analysis/main/Final%20Project%20Files/Data/opposition_comments.csv')
# %%
# Define a function to clean text data
# Function to clean text data
def clean_text(text):
    """
    Cleans the input text by lowercasing, removing URLs, special characters,
    punctuation, numbers, and extra spaces.
    """
    if pd.isna(text):
        return "missing"  # Handle missing values early

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    return text

ruling_df['clean_title_text'] = ruling_df['Title'].apply(clean_text)
ruling_df['clean_text_col'] = ruling_df['Text'].apply(clean_text)


#%%
opposition_df['clean_title_text'] = opposition_df['Title'].apply(clean_text)

#%%
#ruling comments
ruling_comments_df['clean_comment'] = ruling_comments_df['Comment'].apply(clean_text)

#%%
opposition_comments_df['clean_comment'] = opposition_comments_df['Comment'].apply(clean_text)
# %%
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load tokenizer and model specifically for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment", use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

# Create a pipeline with the loaded model and tokenizer
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Applying sentiment analysis and extracting label and score directly
ruling_df['title_sentiment'] = ruling_df['clean_title_text'].apply(lambda text: sentiment_pipeline(text)[0])
ruling_df['sentiment_label'] = ruling_df['title_sentiment'].apply(lambda x: x['label'])
ruling_df['sentiment_score'] = ruling_df['title_sentiment'].apply(lambda x: x['score'])

# Calculating average sentiment score and grouping by label
average_score = ruling_df['sentiment_score'].mean()
average_score_by_label = ruling_df.groupby('sentiment_label')['sentiment_score'].mean()

# Print the average scores
print("Average Sentiment Score:", average_score)
print("Average Score by Label:\n", average_score_by_label)

#%%

# Load the tokenizer specific to your model
tokenizer2 = AutoTokenizer.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")


# Load a second sentiment analysis model for comparison
distilled_sentiment_classifier = pipeline(
    "sentiment-analysis", 
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    tokenizer= tokenizer2,
    truncation=True,
    return_all_scores=False
)

# Apply the second model
ruling_df['title_sentiment2'] = ruling_df['clean_title_text'].apply(lambda text: distilled_sentiment_classifier(text)[0])

#%%
def get_max_sentiment(sentiment):
    return max(sentiment, key=lambda x: x['score'])

ruling_df['highest_sentiment'] = ruling_df['title_sentiment2'].apply(lambda x: get_max_sentiment(x))

ruling_df['sentiment_label2'] = ruling_df['highest_sentiment'].apply(lambda x: x['label'])
ruling_df['sentiment_score2'] = ruling_df['highest_sentiment'].apply(lambda x: x['score'])

# Grouping by label for the second model and calculating mean scores
average_score_by_label2 = ruling_df.groupby('sentiment_label2')['sentiment_score2'].mean()
print("Average Score by Label for Model 2:\n", average_score_by_label2)

#%%
print('Positive posts ',len(ruling_df[ruling_df['sentiment_label2'] == 'positive']))
print('Negative posts ',len(ruling_df[ruling_df['sentiment_label2'] == 'negative']))
print('Neutral posts ',len(ruling_df[ruling_df['sentiment_label2'] == 'neutral']))

# %%
#opposition
opposition_df['title_sentiment'] = opposition_df['clean_title_text'].apply(lambda text: distilled_sentiment_classifier(text)[0])

#%%
# Apply the function to extract the sentiment with the highest score
opposition_df['highest_sentiment'] = opposition_df['title_sentiment'].apply(get_max_sentiment)

# Extract label and score into separate columns
opposition_df['sentiment_label'] = opposition_df['highest_sentiment'].apply(lambda x: x['label'])
opposition_df['sentiment_score'] = opposition_df['highest_sentiment'].apply(lambda x: x['score'])

# Group by label and calculate mean scores for the sentiments
average_score_by_label = opposition_df.groupby('sentiment_label')['sentiment_score'].mean()
print("Average Score by Label:\n", average_score_by_label)
# %%
print('Positive posts ',len(opposition_df[opposition_df['sentiment_label'] == 'positive']))
print('Negative posts ',len(opposition_df[opposition_df['sentiment_label'] == 'negative']))
print('Neutral posts ',len(opposition_df[opposition_df['sentiment_label'] == 'neutral']))
#%%
# Function to split text into smaller chunks
# Function to split text into smaller chunks
def split_text(text, max_length=400):
    tokens = tokenizer.tokenize(text)
    chunks = [' '.join(tokens[i:i + max_length]) for i in range(0, len(tokens), max_length)]
    return chunks

# Function to classify text with optional splitting for long texts
def preprocess_and_classify(text):
    if pd.isna(text):
        return {'label': 'neutral', 'score': 0}  # Handle missing values as neutral

    tokens = tokenizer.tokenize(text)
    if len(tokens) > 400:
        # Split the text into smaller chunks if it exceeds the token limit
        chunks = split_text(text)
        sentiments = []

        for chunk in chunks:
            results = distilled_sentiment_classifier(chunk)
            sentiments.extend(results)  # Collect all sentiment results
        
        # Combine the results (you can adjust the combination logic as needed)
        # combined_sentiment = max(sentiments, key=lambda x: x['score'])
        return sentiments
    else:
        # Process the text directly if it doesn't exceed the token limit
        results = distilled_sentiment_classifier(text)
        # return max(results, key=lambda x: x['score'])
        return results

# Apply the function to your DataFrame
ruling_comments_df['comment_sentiment'] = ruling_comments_df['clean_comment'].apply(preprocess_and_classify)


#%%
# ruling_comments_df['highest_sentiment'] = ruling_comments_df['comment_sentiment'].apply(get_max_sentiment)

# # Extract label and score into separate columns
# ruling_comments_df['sentiment_label'] = ruling_comments_df['highest_sentiment'].apply(lambda x: x['label'])
# ruling_comments_df['sentiment_score'] = ruling_comments_df['highest_sentiment'].apply(lambda x: x['score'])
# %%
'''opposition comments'''
opposition_comments_df['comment_sentiment'] = opposition_comments_df['clean_comment'].apply(preprocess_and_classify)

# %%
from itertools import chain

# Function to flatten and find the highest score sentiment
def extract_highest_score(sentiments):
    if pd.isna(sentiments).any():
        return {'label': 'neutral', 'score': 0}  # Handle missing values as neutral

    # Flatten the nested list
    flattened_sentiments = list(chain.from_iterable(sentiments))

    # Return the sentiment with the highest score
    if flattened_sentiments:
        return max(flattened_sentiments, key=lambda x: x['score'])
    else:
        return {'label': 'neutral', 'score': 0}  # Handle empty lists as neutral

#%%
# Apply the function to your DataFrame
ruling_comments_df['highest_comment_sentiment'] = ruling_comments_df['comment_sentiment'].apply(extract_highest_score)

#%%
opposition_comments_df['highest_comment_sentiment'] = opposition_comments_df['comment_sentiment'].apply(extract_highest_score)

#%%
# Extract label and score into separate columns
ruling_comments_df['sentiment_label'] = ruling_comments_df['highest_comment_sentiment'].apply(lambda x: x['label'])
ruling_comments_df['sentiment_score'] = ruling_comments_df['highest_comment_sentiment'].apply(lambda x: x['score'])
# %%
# Extract label and score into separate columns
opposition_comments_df['sentiment_label'] = opposition_comments_df['highest_comment_sentiment'].apply(lambda x: x['label'])
opposition_comments_df['sentiment_score'] = opposition_comments_df['highest_comment_sentiment'].apply(lambda x: x['score'])
# %%
ruling_df_csv = ruling_df.copy()
ruling_comments_df_csv = ruling_comments_df.copy()
opposition_df_csv = opposition_df.copy()
opposition_comments_df_csv = opposition_comments_df.copy()
# %%
ruling_df_csv.to_csv('D:/STUDY/MS/Build Project/data scraped/ruling_party_posts_clean.csv', index=False)
ruling_comments_df_csv.to_csv('D:/STUDY/MS/Build Project/data scraped/ruling_comments_clean.csv', index=False)
opposition_df_csv.to_csv('D:/STUDY/MS/Build Project/data scraped/opposition_party_posts_clean.csv', index=False)
opposition_comments_df_csv.to_csv('D:/STUDY/MS/Build Project/data scraped/opposition_comments_clean.csv', index=False)