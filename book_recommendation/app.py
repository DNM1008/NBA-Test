import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# Load data
books = pd.read_csv("Books.csv")
ratings = pd.read_csv("Ratings.csv")
users = pd.read_csv("Users.csv")

ratings_with_name = ratings.merge(books, on="ISBN")

# Filter ratings for popular books and active users
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
book_reads = x[x].index

filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(book_reads)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

# Create pivot table
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

# Compute similarity scores
similarity_scores = cosine_distances(pt)

# Define recommendation function
def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=False)[1:6]
    recommendations = [pt.index[i[0]] for i in similar_items]
    return recommendations

# Streamlit app
st.title("Book Recommendation System")
st.write("Select a book from the dropdown menu to get recommendations.")

# Dropdown menu for book selection
book_list = pt.index.tolist()
selected_book = st.selectbox("Choose a book:", book_list)

if st.button("Recommend"):
    if selected_book:
        recommendations = recommend(selected_book)
        st.write("**Recommended books:**")
        for book in recommendations:
            st.write(f"- {book}")
    else:
        st.write("Please select a book from the dropdown.")
