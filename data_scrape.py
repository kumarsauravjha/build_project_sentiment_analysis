#%%
'''working draft'''
import pandas as pd
import numpy as np
import praw

#%%
reddit = praw.Reddit(
    client_id = 'QEGO8TfQnhOw69Lrn1moQQ',
    client_secret = 'KWsq5C70fuix19vUzh0zW4plnx2efA',
    user_agent = 'my-app by u/AloneAbies1630'
)

# %%
# # Function to search for posts containing specific keywords
# def search_reddit(query, limit=100):
#     posts = reddit.subreddit('all').search(query, limit=limit)
#     post_list = [[post.title, post.selftext, post.score, post.subreddit.display_name, post.url] for post in posts]
#     return pd.DataFrame(post_list, columns=['Title', 'Text', 'Score', 'Subreddit', 'URL'])

# # Search for posts related to Indian General Elections 2024
# query = "Indian General Elections 2024"
# search_results = search_reddit(query, limit=1000)

# # Save to CSV
# # search_results.to_csv('reddit_search_results.csv', index=False)

# # print("Search results saved to reddit_search_results.csv")


# %%
from datetime import datetime
# Function to search for posts containing specific keywords
def search_reddit(query, start_date, end_date, limit=100):
    posts = reddit.subreddit('all').search(query, limit=limit)
    post_list = []
    
    for post in posts:
        post_date = datetime.fromtimestamp(post.created_utc)
        if start_date <= post_date <= end_date:
            # Including more metadata and permalink
            post_list.append([
                post.id,
                post.title,
                post.selftext,
                post.score,
                post.num_comments,
                post.upvote_ratio,
                post.subreddit.display_name,
                f"https://www.reddit.com{post.permalink}",
                post.url,
                post_date
            ])
    
    return pd.DataFrame(post_list, columns=[
        'Post_ID', 'Title', 'Text', 'Score', 'Num_Comments', 'Upvote_Ratio', 'Subreddit', 'Permalink', 'URL', 'Date'
    ])
# Specify the date range
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 5, 31)

# List of keywords
# keywords_ruling_party = [
#     "Indian General Elections 2024",
#     "Indian elections 2024",
#     "India elections 2024",
#     "Lok Sabha elections 2024",
#     "General elections India 2024", 
#     "Narendra Modi", 
#     "Bhartiya Janta Party", 
#     "BJP",
    
# ]

keywords_ruling_party = ["Narendra Modi", "Bharatiya Janata Party", "BJP", "Amit Shah" ]

keywords_opposition = ["Indian National Congress" , "INC", "Rahul Gandhi", "INDIA alliance", "Sonia Gandhi", "INDI alliance" ]
# Initialize an empty DataFrame
ruling_df = pd.DataFrame(columns=['Title', 'Text', 'Score', 'Subreddit', 'URL'])
opposition_df = pd.DataFrame(columns=['Title', 'Text', 'Score', 'Subreddit', 'URL'])

# Search for posts using each keyword and combine results
search_limit = 1000
for keyword in keywords_ruling_party:
    search_results_df_ruling = search_reddit(keyword, start_date, end_date, limit=search_limit)
    ruling_df = pd.concat([ruling_df, search_results_df_ruling], ignore_index=True)

# Remove duplicates
ruling_df.drop_duplicates(subset=['Title', 'Text', 'URL'], inplace=True)

for keyword in keywords_opposition:
    search_results_df_opposition = search_reddit(keyword, start_date, end_date, limit=search_limit)
    opposition_df = pd.concat([opposition_df, search_results_df_opposition], ignore_index=True)

opposition_df.drop_duplicates(subset=['Title', 'Text', 'URL'], inplace=True)

# Display the combined DataFrame
# print(all_search_results_df)
print(f"Total posts for BJP {len(ruling_df)}")
print(f"Total posts for INC", len(opposition_df))
#%%
ruling_df.to_csv("D:/STUDY/MS/Build Project/ruling_party_posts.csv")
opposition_df.to_csv("D:/STUDY/MS/Build Project/opposition_posts.csv")
# %%
#now we'll pull the comments from all the posts
# Function to fetch comments from a given post ID
# def fetch_comments(post_id):
#     submission = reddit.submission(id=post_id)
#     submission.comments.replace_more(limit=None)  # This line ensures you get all comments, not just the top level
#     comments = []
#     for comment in submission.comments.list():
#         comments.append([submission.id, comment.id, comment.body, comment.score, comment.created_utc])
#     return pd.DataFrame(comments, columns=['Post_ID', 'Comment_ID', 'Comment', 'Score', 'Created'])

# # Apply the function to each DataFrame and store the results
# def get_comments_for_df(df):
#     # Extract post IDs from URLs
#     df['Post_ID'] = df['URL'].apply(lambda x: x.split('/')[-3])
#     comments_df = pd.DataFrame(columns=['Post_ID', 'Comment_ID', 'Comment', 'Score', 'Created'])
#     for post_id in df['Post_ID'].unique():
#         post_comments_df = fetch_comments(post_id)
#         comments_df = pd.concat([comments_df, post_comments_df], ignore_index=True)
#     return comments_df

# # Get comments for both dataframes
# ruling_comments_df = get_comments_for_df(ruling_df)
# opposition_comments_df = get_comments_for_df(opposition_df)

# Optionally, save to CSV
# ruling_comments_df.to_csv('ruling_comments.csv', index=False)
# opposition_comments_df.to_csv('opposition_comments.csv', index=False)

# # Display some information about the comments collected
# print(f"Total comments for ruling party: {len(ruling_comments_df)}")
# print(f"Total comments for opposition: {len(opposition_comments_df)}")
# %%

# Function to fetch comments from a given post ID
def fetch_comments(post_id):
    if not post_id:  # Check if the post_id is empty or None
        print("Invalid post ID encountered.")
        return pd.DataFrame(columns=['Post_ID', 'Comment_ID', 'Comment', 'Score', 'Created'])
    try:
        submission = reddit.submission(id=post_id)
        submission.comments.replace_more(limit=None)
        comments = []
        for comment in submission.comments.list():
            comments.append([
                submission.id, comment.id, comment.body, comment.score, comment.created_utc
            ])
        return pd.DataFrame(comments, columns=['Post_ID', 'Comment_ID', 'Comment', 'Score', 'Created'])
    except Exception as e:
        print(f"Error fetching comments for post ID {post_id}: {e}")
        return pd.DataFrame(columns=['Post_ID', 'Comment_ID', 'Comment', 'Score', 'Created'])

# Function to process DataFrame and fetch comments for all posts
def get_comments_for_df(df):
    comments_df = pd.DataFrame(columns=['Post_ID', 'Comment_ID', 'Comment', 'Score', 'Created'])
    for post_id in df['Post_ID'].unique():
        post_comments_df = fetch_comments(post_id)
        comments_df = pd.concat([comments_df, post_comments_df], ignore_index=True)
    return comments_df

# Assuming ruling_df and opposition_df are already loaded with a 'Post_ID' column
# Get comments for both dataframes
ruling_comments_df = get_comments_for_df(ruling_df)
opposition_comments_df = get_comments_for_df(opposition_df)

# # Optionally, save to CSV
# ruling_comments_df.to_csv('ruling_comments.csv', index=False)
# opposition_comments_df.to_csv('opposition_comments.csv', index=False)

# Display some information about the comments collected
print(f"Total comments for ruling party: {len(ruling_comments_df)}")
print(f"Total comments for opposition: {len(opposition_comments_df)}")
# %%
# Optionally, save to CSV
ruling_comments_df.to_csv('D:/STUDY/MS/Build Project/data scraped/ruling_comments.csv', index=False)
opposition_comments_df.to_csv('D:/STUDY/MS/Build Project/data scraped/opposition_comments.csv', index=False)

#%%
ruling_df.to_csv('D:/STUDY/MS/Build Project/data scraped/ruling_party_posts.csv', index=False)
opposition_df.to_csv('D:/STUDY/MS/Build Project/data scraped/opposition_party_posts.csv', index=False)
# %%
