#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
import spacy
from nltk.corpus import stopwords
#%%
#data import
ruling_df = pd.read_csv('https://raw.githubusercontent.com/kumarsauravjha/build_project_sentiment_analysis/main/Final%20Project%20Files/Data/ruling_party_posts_clean.csv')
ruling_comments_df= pd.read_csv('https://raw.githubusercontent.com/kumarsauravjha/build_project_sentiment_analysis/main/Final%20Project%20Files/Data/ruling_comments_clean.csv')
opposition_df = pd.read_csv('https://raw.githubusercontent.com/kumarsauravjha/build_project_sentiment_analysis/main/Final%20Project%20Files/Data/opposition_party_posts_clean.csv')
opposition_comments_df = pd.read_csv('D:/STUDY/MS/Build Project/data scraped/opposition_comments_clean2.csv')

# %%
# Pie Chart for Sentiment Distribution
def plot_sentiment_distribution(df, column, title):
    sentiment_counts = df[column].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.show()

plot_sentiment_distribution(ruling_comments_df, 'sentiment_label', 'Sentiment Distribution for Ruling Party Comments')
plot_sentiment_distribution(opposition_comments_df, 'sentiment_label', 'Sentiment Distribution for Opposition Party Comments')

plot_sentiment_distribution(ruling_df, 'sentiment_label', 'Sentiment Distribution for Ruling Party Posts')
plot_sentiment_distribution(opposition_df, 'sentiment_label', 'Sentiment Distribution for Opposition Party Posts')

# %%
# Line Chart for Sentiment Over Time
def plot_sentiment_over_time(df, date_column, sentiment_column):
    plt.figure(figsize=(12, 6))
    df.groupby(date_column)[sentiment_column].mean().plot()
    plt.title('Average Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.show()

plot_sentiment_over_time(ruling_comments_df, 'Created', 'sentiment_score')

# %%
# Bar Chart for Sentiment by Keyword
def plot_sentiment_by_keyword(df, keyword_column, sentiment_column):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=keyword_column, y=sentiment_column, data=df)
    plt.title('Sentiment by Keyword')
    plt.xlabel('Keyword')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=45)
    plt.show()

plot_sentiment_by_keyword(ruling_comments_df, 'Keyword', 'sentiment_score')

# %%

plt.figure(figsize=(10,8))
sns.barplot(x=['Ruling Party Comments', 'Opposition Comments'], y=[len(ruling_comments_df), len(opposition_comments_df)], color=)
plt.grid(True)
plt.show()
# %%
# Combine all the clean text data into a single string
ruling_positive=ruling_comments_df[ruling_comments_df['sentiment_label'] == 'positive']
ruling_negative=ruling_comments_df[ruling_comments_df['sentiment_label'] == 'negative']
#%%
all_text_ruling_comments_positive = ' '.join(ruling_positive['cleaned_comment'].dropna())
all_text_ruling_comments_negative = ' '.join(ruling_negative['cleaned_comment'].dropna())

# Generate the word cloud
wordcloud1 = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(all_text_ruling_comments_positive)

wordcloud2 = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(all_text_ruling_comments_negative)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.title('Word Cloud for Ruling Party Positive', fontsize=20)
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.title('Word Cloud for Ruling Party Negative', fontsize=20)
plt.show()

# %%
'''stopwords removal and lemmatization'''
# Download NLTK stopwords
nltk.download('stopwords')
# %%
# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# Function to clean text data
def clean_text(text):
    if pd.isna(text):
        return "missing"  # Handle missing values early

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in text.lower().split() if token not in stop_words]

    # Lemmatize
    doc = nlp(' '.join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]

    return ' '.join(lemmatized_tokens)
# %%
# Apply the cleaning function to your DataFrame
ruling_comments_df['cleaned_comment'] = ruling_comments_df['clean_comment'].apply(clean_text)

# %%
opposition_comments_df['cleaned_comment'] = opposition_comments_df['clean_comment'].apply(clean_text)

# %%
# Combine all the clean text data into a single string
opposition_positive=opposition_comments_df[opposition_comments_df['sentiment_label'] == 'positive']
opposition_negative=opposition_comments_df[opposition_comments_df['sentiment_label'] == 'negative']

all_text_opposition_comments_positive = ' '.join(opposition_positive['cleaned_comment'].dropna())
all_text_opposition_comments_negative = ' '.join(opposition_negative['cleaned_comment'].dropna())

# Generate the word cloud
wordcloud3 = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(all_text_opposition_comments_positive)

wordcloud4 = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(all_text_opposition_comments_negative)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud3, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.title('Word Cloud for Opposition Positive', fontsize=20)
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud4, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.title('Word Cloud for Opposition Negative', fontsize=20)
plt.show()
# %%

# Calculate average Reddit scores for posts and comments
average_scores = {
    'Party': ['Ruling', 'Ruling', 'Opposition', 'Opposition'],
    'Type': ['Posts', 'Comments', 'Posts', 'Comments'],
    'Average_Score': [
        ruling_df['Score'].mean(),
        ruling_comments_df['Score'].mean(),
        opposition_df['Score'].mean(),
        opposition_comments_df['Score'].mean()
    ]
}

average_scores_df = pd.DataFrame(average_scores)


# %%
# Data for posts
average_scores_posts = average_scores_df[average_scores_df['Type'] == 'Posts']
# Data for comments
average_scores_comments = average_scores_df[average_scores_df['Type'] == 'Comments']

# Create subplots
plt.figure(figsize=(14, 6))

# Plot the counts of posts for each party
plt.subplot(1, 2, 1)
# Plot for posts
# plt.figure(figsize=(10, 6))
sns.barplot(x='Party', y='Average_Score', data=average_scores_posts)
plt.title('Average Reddit Scores for Posts')
plt.ylabel('Average Reddit Score')
plt.xlabel('Party')
plt.grid(axis='y')

plt.subplot(1, 2, 2)
# Plot for comments
# plt.figure(figsize=(10, 6))
sns.barplot(x='Party', y='Average_Score', data=average_scores_comments)
plt.title('Average Reddit Scores for Comments')
plt.ylabel('Average Reddit Score')
plt.xlabel('Party')
plt.grid(axis='y')
# plt.show()

# Display the plots
plt.tight_layout()
# plt.grid(axis='y')
plt.show()


# %%

# Calculate the average number of comments for each party
ruling_avg_comments = ruling_df['Num_Comments'].mean()
opposition_avg_comments = opposition_df['Num_Comments'].mean()

# Create a DataFrame for the average number of comments
avg_comments_df = pd.DataFrame({
    'Party': ['Ruling', 'Opposition'],
    'Average_Num_Comments': [ruling_avg_comments, opposition_avg_comments]
})

# Plot the average number of comments for each party
plt.figure(figsize=(10, 6))
sns.barplot(x='Party', y='Average_Num_Comments', data=avg_comments_df)
plt.title('Average Number of Comments for Posts')
plt.ylabel('Average Number of Comments')
plt.xlabel('Party')
plt.grid(axis='y')
plt.show()
# %%
# Calculate the number of posts and comments for each party
ruling_posts_count = len(ruling_df)
ruling_comments_count = len(ruling_comments_df)
opposition_posts_count = len(opposition_df)
opposition_comments_count = len(opposition_comments_df)

# Create DataFrames for posts and comments
posts_counts_df = pd.DataFrame({
    'Party': ['Ruling', 'Opposition'],
    'Count': [ruling_posts_count, opposition_posts_count]
})

comments_counts_df = pd.DataFrame({
    'Party': ['Ruling', 'Opposition'],
    'Count': [ruling_comments_count, opposition_comments_count]
})

# Create subplots
plt.figure(figsize=(14, 6))

# Plot the counts of posts for each party
plt.subplot(1, 2, 1)
sns.barplot(x='Party', y='Count', data=posts_counts_df)
plt.title('Number of Posts for Ruling and Opposition Parties')
plt.ylabel('Number of Posts')
plt.xlabel('Party')
plt.grid(axis='y')

# Plot the counts of comments for each party
plt.subplot(1, 2, 2)
sns.barplot(x='Party', y='Count', data=comments_counts_df)
plt.title('Number of Comments for Ruling and Opposition Parties')
plt.ylabel('Number of Comments')
plt.xlabel('Party')

# Display the plots
plt.tight_layout()
plt.grid(axis='y')
plt.show()


# %%
def plot_sentiment_distribution2(df, column, title):
    sentiment_counts = df[column].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.show()

# Assuming 'sentiment_label' column exists in both dataframes
plot_sentiment_distribution2(ruling_df, 'sentiment_label2', 'Sentiment Distribution for Ruling Party Posts')
plot_sentiment_distribution2(opposition_df, 'sentiment_label', 'Sentiment Distribution for Opposition Party Posts')

# %%
ruling_df['sentiment_label2'].value_counts()

# %%
opposition_df['sentiment_label'].value_counts()
# %%
