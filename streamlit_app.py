import streamlit as st
import pandas as pd
import numpy as np
import torch, utils

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🍿",
    layout="wide"
)

@st.cache_data
def load_ratings():
    #Ratings
    ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, encoding='latin-1', engine='python')
    ratings.columns = ['userId','movieId','rating','timestamp']
    return ratings


@st.cache_data
def load_movies():
    #Movies
    movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, encoding='latin-1', engine='python')
    movies.columns = ['movieId','title','genres']
    return movies

@st.cache_data
def load_users():
    #Users
    users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, encoding='latin-1',  engine='python')
    users.columns = ['userId','gender','age','occupation','zipCode']
    return users

ratings = load_ratings()
movies = load_movies()
users = load_users()


def preprocessed_data(ratings, users, movies):
    # movie preprocessing
    movie_records = movies.copy()
    movies['genres'] = movies.apply(lambda row : row['genres'].split("|")[0],axis=1)
    movies['movie_year'] = movies.apply(lambda row : int(row['title'].split("(")[-1][:-1]),axis=1)
    movies.drop(['title'],axis=1,inplace=True)

    # combine rating and movie
    rating_movie = pd.merge(ratings,movies,how='left',on="movieId")

    # user preprocessing
    pd.set_option('future.no_silent_downcasting', True)
    users['gender'] = users['gender'].replace({'F':0,'M':1}).astype('int64')
    users['age'] = users['age'].replace({1:0,18:1, 25:2, 35:3, 45:4, 50:5, 56:6 })
    users.drop(['zipCode'],axis=1,inplace=True)

    # combine into final dataframe
    final_df = pd.merge(rating_movie,users,how='left',on='userId')

    return final_df, movie_records

final_df, movie_records = preprocessed_data(ratings, users, movies)

#settings for the data
wide_cols = ['movie_year','gender','age', 'occupation','genres','userId','movieId']
embeddings_cols = [('genres',20), ('userId',100), ('movieId',100)]
continuous_cols = ["movie_year","gender","age","occupation"]
target = 'rating'

#split the data and generate the embeddings
def data_process(final_df, wide_cols, embeddings_cols, continuous_cols, target):
    data_processed = utils.data_processing(final_df, wide_cols, embeddings_cols, continuous_cols, target, scale=True)
    return data_processed

data_processed = data_process(final_df, wide_cols, embeddings_cols, continuous_cols, target)

#setup for model arguments
wide_dim = data_processed['train_dataset'].wide.shape[1]
deep_column_idx = data_processed['deep_column_idx']
embeddings_input= data_processed['embeddings_input']
encoding_dict   = data_processed['encoding_dict']
hidden_layers = [100,50]
dropout = [0.5,0.5]

model_args = {
    'wide_dim': wide_dim,
    'embeddings_input': embeddings_input,
    'continuous_cols': continuous_cols,
    'deep_column_idx': deep_column_idx,
    'hidden_layers': hidden_layers,
    'dropout': dropout,
    'encoding_dict': encoding_dict,  # Will be updated during loading
    'n_class': 1
}

@st.cache_resource
def load_models(model_args):
    loaded_model = utils.load_model(utils.NeuralNet, model_args, "./model/movie_recommendation_model.pth", utils.device)
    loaded_model.compile(optimizer='Adam')
    return  loaded_model

loaded_model = load_models(model_args)

#predict_user = 1000

#top_k_movies = utils.recommend_top_k_movies(predict_user, final_df, movie_records, loaded_model, wide_cols, embeddings_cols, continuous_cols, k = 10, search_term = None)




# Streamlit UI Setup
st.title("Movie Recommendation System")

# Streamlit Tabs
tabs = st.tabs(["Home", "Dashboard", "Documentation"])

with tabs[0]:
    # Home Tab: The introduction and user inputs for recommendations
    st.header("Welcome to the Movie Recommendation System")
    
    st.markdown("""
    This app recommends movies to users based on their previous interactions. 
    You can enter your user ID, search for specific movies by title or genre, 
    and get personalized movie recommendations. Choose the number of recommendations you want to receive.
    """)

    # User selection: Enter user ID
    user_id = st.number_input("Enter User ID", min_value=1, max_value=final_df['userId'].max(), value=1)

    # Search feature: User can enter a query
    search_term = st.text_input("Search for Movies (Title or Genre)", "")

    # Get movie recommendations
    k = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)
    recommended_movies = utils.recommend_top_k_movies(
        predict_user=user_id,
        final_df=final_df,
        movie_records=movie_records,
        model=loaded_model,  # Replace with your actual model
        wide_cols=wide_cols,
        embeddings_cols=embeddings_cols,
        continuous_cols=continuous_cols,
        search_term=search_term
    ).head(k)

    # Display Recommendations
    st.subheader(f"Top {k} Movie Recommendations for User {user_id}")
    if not recommended_movies.empty:
        # Add "No." column and set it as index
        recommended_movies["No."] = range(1, len(recommended_movies) + 1)
        recommended_movies = recommended_movies.set_index("No.")
        recommended_movies["rating"] = 0
        recommended_movies["comments"] = ""
        st.data_editor(recommended_movies[['title', 'genres', 'movie_year', 'rating', 'comments']],
                      column_config={
                        "rating": st.column_config.NumberColumn(
                            "Your rating",
                            help="How much do you like this movie (1-5)?",
                            min_value=1,
                            max_value=5,
                            step=1,
                            format="%d ⭐",
                        ),
                        "comments": "Comments",
                    },
                    disabled=["title", "genres", "movie_year"],
                    use_container_width = True
                )

        # Combine all the text from the DataFrame into a single string
        text = ' '.join(recommended_movies['genres'])
        with st.container(border=True):
            # Add a button to enable/disable word cloud
            if st.button('Generate Word Cloud'):
                st.write("The word cloud is now visible!")
                st.pyplot(utils.generate_wordcloud(text))
            else:
                st.write("Click the button above to generate the word cloud.")
    else:
        st.write("No recommendations available for this user with the search query.")

with tabs[1]:
    # Dashboard Tab: Data visualization and exploration
    st.header("Data Dashboard")
    st.markdown("""
    This section provides visual insights into the movie dataset. 
    You can explore distribution of ratings, genres, and other useful metrics.
    """)

    # Genre Distribution
    st.subheader("Genre Distribution")
    genre_counts = movie_records['genres'].value_counts()
    st.bar_chart(genre_counts)

    # Ratings Distribution
    st.subheader("Ratings Distribution")
    rating_counts = final_df['rating'].value_counts().sort_index()
    st.line_chart(rating_counts)

    # Movie Year Distribution
    st.subheader("Movie Release Year Distribution")
    year_counts = movie_records['movie_year'].value_counts().sort_index()
    st.line_chart(year_counts)

    # Correlation between rating and movie year (example)
    st.subheader("Correlation between Movie Year and Rating")
    rating_by_year = final_df.groupby('movieId')['rating'].mean().reset_index()
    movie_year_ratings = rating_by_year.merge(movie_records[['movieId', 'movie_year']], on='movieId')
    movie_year_ratings = movie_year_ratings.groupby('movie_year')['rating'].mean().reset_index()
    st.line_chart(movie_year_ratings.set_index('movie_year'))

with tabs[2]:
    # Documentation Tab: Information about the system
    st.header("Documentation")
    st.markdown("""
    ## Overview
    The Movie Recommendation System uses collaborative filtering and personalized recommendations based on user behavior.

    **Github Link**: https://github.com/FUZHANGCHENG-23071497/e-commerce_recom_system/tree/main
    
    ## How it Works
    - **User Interaction**: The app uses historical user interactions (ratings) and movie data to recommend movies.
    - **Model**: The model predicts ratings for unrated movies, and top-k recommendations are generated for each user.

    ## Recommendation Algorithm
    The Movie Recommendation System uses a **Wide & Deep** learning model for collaborative filtering.
    - **Wide Model**: Captures interactions between features (e.g., user and movie ID) that may not have been seen in the training data. This model leverages cross-product features to model interactions, ensuring that new or rare combinations of features are still considered.
    - **Deep Model**: Uses deep neural networks to capture complex patterns and relationships between the input features (e.g., genre, ratings, time of interaction). The deep model is excellent for identifying latent factors and understanding complex interactions between users and movies.
    By combining these two models, the system can provide both general recommendations (wide model) and more personalized suggestions (deep model), improving the quality of recommendations over time.
    
    ## Features:
    - **User Input**: Allows users to enter their ID and preferences.
    - **Search Function**: Search movies by title or genre.
    - **Personalized Recommendations**: Get top-k movie recommendations based on predicted ratings.
    - **Data Visualizations**: Explore movie genre distributions, rating trends, and more.

    ## Feedback
    We value your feedback to improve our system! Please share your thoughts and suggestions.
    """)

    # Add a user feedback form
    st.subheader("Share Your Feedback")
    st.markdown("We'd love to hear from you! Please let us know your thoughts, suggestions, or any issues you've encountered.")
    
    # Create feedback form inputs
    user_name = st.text_input("Name (optional):")
    user_email = st.text_input("Email (optional):")
    feedback = st.text_area("Your Feedback:")
    
    # Submit button
    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback!")
            # Here you can integrate code to save feedback to a database or send it via email
        else:
            st.error("Feedback cannot be empty. Please share your thoughts.")
    
    # FAQ Section
    st.subheader("Frequently Asked Questions (FAQ)")
    st.markdown("""
    **1. How does the system recommend movies?**  
    The system uses a combination of collaborative filtering and the Wide & Deep model to predict user preferences and generate recommendations based on past interactions and movie features.
    
    **2. What kind of data does the system use?**  
    The system uses user ratings, movie metadata (e.g., genres, release year), and interaction history to train the model.
    
    **3. Can I search for specific movies?**  
    Yes! Use the search function to look for movies by title or genre.
    
    **4. How are my preferences updated?**  
    The system learns from your interactions (e.g., ratings, searches) and adjusts recommendations dynamically over time.
    
    **5. Is my feedback anonymous?**  
    Yes, unless you choose to provide your name or email in the feedback form, your feedback is anonymous.
    
    **6. Can I request new features or report issues?**  
    Absolutely! Use the feedback form above to share your suggestions or report any issues you've encountered.
    """)
