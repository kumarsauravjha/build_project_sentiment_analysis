#%%
'''working draft'''
import pandas as pd
import numpy as np


# %%
import praw

#%%
reddit = praw.Reddit(
    client_id = 'QEGO8TfQnhOw69Lrn1moQQ',
    client_secret = 'KWsq5C70fuix19vUzh0zW4plnx2efA',
    user_agent = 'my-app by u/AloneAbies1630'
)

india_speaks = reddit.subreddit("IndiaSpeaks")

india = reddit.subreddit("india")

top_posts = india_speaks.top(limit=10)
new_posts = india_speaks.new(limit=10)
# %%
for post in top_posts:
    print(f"Title- {post.title}")
# %%
# Function to search for posts containing specific keywords
def search_reddit(query, limit=100):
    posts = reddit.subreddit('all').search(query, limit=limit)
    post_list = [[post.title, post.selftext, post.score, post.subreddit.display_name, post.url] for post in posts]
    return pd.DataFrame(post_list, columns=['Title', 'Text', 'Score', 'Subreddit', 'URL'])

# Search for posts related to Indian General Elections 2024
query = "Indian General Elections 2024"
search_results = search_reddit(query, limit=1000)

# Save to CSV
# search_results.to_csv('reddit_search_results.csv', index=False)

# print("Search results saved to reddit_search_results.csv")


# %%
from datetime import datetime
# Function to search for posts containing specific keywords
def search_reddit(query, start_date, end_date, limit=100):
    posts = reddit.subreddit('all').search(query, limit=limit)
    post_list = []
    
    for post in posts:
        post_date = datetime.fromtimestamp(post.created_utc)
        if start_date <= post_date <= end_date:
            post_list.append([post.title, post.selftext, post.score, post.subreddit.display_name, post.url, post_date])
    
    return pd.DataFrame(post_list, columns=['Title', 'Text', 'Score', 'Subreddit', 'URL', 'Date'])

# Specify the date range
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 5, 31)

# List of keywords
keywords = [
    "Indian General Elections 2024",
    "Indian elections 2024",
    "India elections 2024",
    "Lok Sabha elections 2024",
    "General elections India 2024"
]

# Initialize an empty DataFrame
all_search_results_df = pd.DataFrame(columns=['Title', 'Text', 'Score', 'Subreddit', 'URL'])

# Search for posts using each keyword and combine results
search_limit = 1000
for keyword in keywords:
    search_results_df = search_reddit(keyword, start_date, end_date, limit=search_limit)
    all_search_results_df = pd.concat([all_search_results_df, search_results_df], ignore_index=True)

# Remove duplicates
all_search_results_df.drop_duplicates(subset=['Title', 'Text', 'URL'], inplace=True)

# Display the combined DataFrame
print(all_search_results_df)
print(f"Total posts retrieved: {len(all_search_results_df)}")
# %%
