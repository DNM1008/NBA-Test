import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances

books = pd.read_csv("Books.csv")
ratings = pd.read_csv("Ratings.csv")
users = pd.read_csv("Users.csv")

ratings_with_name = ratings.merge(books,on = "ISBN")

# Group by 'Book-Title' and count 'Book-Rating'
num_rating_df_t = ratings_with_name.groupby('Book-Title').count()['Book-Rating']

# Sort the DataFrame in descending order
num_rating_df_t = num_rating_df_t.sort_values(ascending=False).reset_index()
num_rating_df_t.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
num_rating_df_t
# Display the result
num_rating_df_t

ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')

num_rating_df_t = ratings_with_name.groupby('Book-Title').mean(numeric_only=True)['Book-Rating']

num_rating_df_t = num_rating_df_t.to_frame()
num_rating_df_t.rename(columns={'Book-Rating':'avg_rating'}, inplace=True)
num_rating_df_t = num_rating_df_t.reset_index()

num_rating_df_t

# Group by 'Book-Title' and count 'Book-Rating'
num_rating_df_t = ratings_with_name.groupby('Book-Title').agg({'Book-Rating': 'count'}).reset_index()

# Sort the DataFrame in descending order of ratings
num_rating_df_t = num_rating_df_t.sort_values(by='Book-Rating', ascending=False).reset_index(drop=True)

# Group by 'Book-Author' and count 'Book-Rating'
num_rating_df = ratings_with_name.groupby('Book-Author').agg({'Book-Rating': 'count'}).reset_index()

# Sort the DataFrame in descending order of ratings
num_rating_df = num_rating_df.sort_values(by='Book-Author', ascending=False).reset_index(drop=True)

num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
num_rating_df

valid_ratings = ratings_with_name[pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce').notna()]
avg_rating_df = valid_ratings.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_rating'}, inplace=True)
avg_rating_df

popular_df = num_rating_df.merge(avg_rating_df,on='Book-Title')
popular_df


popular_df = popular_df[popular_df['num_ratings']>=250].sort_values('avg_rating',ascending=False).head(60)
popular_df = popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_rating']]



x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
book_reads = x[x].index


filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(book_reads)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
final_ratings

pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')

pt.fillna(0,inplace=True)

similarity_scores = cosine_distances(pt)

def recommend(book_name):
    # index fetch
    index = np.where(pt.index == book_name )[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key = lambda x:x[1],reverse = True )[1:6]
    for i in similar_items:
        print(pt.index[i[0]])
    
