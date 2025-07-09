import pandas as pd
import numpy as np
import torch

from collections import namedtuple
from itertools import chain
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encoder(df, cols=None):
    if cols == None:
        cols = list(df.select_dtypes(include=['object']).columns)

    val_types = dict()
    for c in cols:
        val_types[c] = df[c].unique()

    val_to_idx = dict()
    for k, v in val_types.items():
        val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

    for k, v in val_to_idx.items():
        df[k] = df[k].apply(lambda x: v[x])

    return val_to_idx, df

#function used to break the data into test and train sets and make embedding of the data
def data_processing(df, wide_cols, embeddings_cols, continuous_cols, target,
    scale=False, def_dim=8):


    if type(embeddings_cols[0]) is tuple:
        emb_dim = dict(embeddings_cols)
        embeddings_cols = [emb[0] for emb in embeddings_cols]
    else:
        emb_dim = {e:def_dim for e in embeddings_cols}
    deep_cols = embeddings_cols+continuous_cols

    # Extract the target and copy the dataframe so we don't mutate it
    # internally.
    Y = np.array(df[target])
    all_columns = list(set(wide_cols + deep_cols ))
    df_tmp = df.copy()[all_columns]


    # Extract the categorical column names that can be one hot encoded later
    categorical_columns = list(df_tmp.select_dtypes(include=['object']).columns)

    
    encoding_dict,df_tmp = encoder(df_tmp)
    encoding_dict = {k:encoding_dict[k] for k in encoding_dict if k in deep_cols}
    embeddings_input = []
    for k,v in encoding_dict.items():
        embeddings_input.append((k, len(v), emb_dim[k]))

    df_deep = df_tmp[deep_cols]
    deep_column_idx = {k:v for v,k in enumerate(df_deep.columns)}


    if scale:
        scaler = StandardScaler()
        for cc in continuous_cols:
            df_deep[cc]  = scaler.fit_transform(df_deep[cc].values.reshape(-1,1))

    df_wide = df_tmp[wide_cols]
    del(df_tmp)
    dummy_cols = [c for c in wide_cols if c in categorical_columns]
    df_wide = pd.get_dummies(df_wide, columns=dummy_cols)

    X_train_deep, X_test_deep = train_test_split(df_deep.values, test_size=0.3, random_state=1463)
    X_train_wide, X_test_wide = train_test_split(df_wide.values, test_size=0.3, random_state=1463)
    y_train, y_test = train_test_split(Y, test_size=0.3, random_state=1981)

    group_dataset = dict()
    train_dataset = namedtuple('train_dataset', 'wide, deep, labels')
    test_dataset  = namedtuple('test_dataset' , 'wide, deep, labels')
    group_dataset['train_dataset'] = train_dataset(X_train_wide, X_train_deep, y_train)
    group_dataset['test_dataset']  = test_dataset(X_test_wide, X_test_deep, y_test)
    group_dataset['embeddings_input']  = embeddings_input
    group_dataset['deep_column_idx'] = deep_column_idx
    group_dataset['encoding_dict'] = encoding_dict

    return group_dataset

#Loadthe dataset
class DatasetLoader(Dataset):
    def __init__(self, data):

        # Access the tuple fields directly, not by string indexing
        self.X_wide = np.array(data.wide, dtype=np.float32)
        self.X_deep = np.array(data.deep, dtype=np.float32)
        self.Y = np.array(data.labels, dtype=np.float32)

    def __getitem__(self, idx):

        xw = torch.tensor(self.X_wide[idx], dtype=torch.float64)
        xd = torch.tensor(self.X_deep[idx], dtype=torch.float64)
        y = torch.tensor(self.Y[idx], dtype=torch.float64)

        return xw, xd, y

    def __len__(self):
        return len(self.Y)
    
#class defining the wide and deep neural network
class NeuralNet(nn.Module):

    def __init__(self,
                 wide_dim,
                 embeddings_input,
                 continuous_cols,
                 deep_column_idx,
                 hidden_layers,
                 dropout,
                 encoding_dict,
                 n_class):

        super(NeuralNet, self).__init__()
        self.wide_dim = wide_dim
        self.deep_column_idx = deep_column_idx
        self.embeddings_input = embeddings_input
        self.continuous_cols = continuous_cols
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.encoding_dict = encoding_dict
        self.n_class = n_class
        self.loss_values=[]

        # Build the embedding layers to be passed through the deep-side
        for col,val,dim in self.embeddings_input:
            setattr(self, 'emb_layer_'+col, nn.Embedding(val, dim))

        # Build the deep-side hidden layers with dropout if specified
        input_emb_dim = np.sum([emb[2] for emb in self.embeddings_input])
        self.linear_1 = nn.Linear(input_emb_dim+len(continuous_cols), self.hidden_layers[0])
        if self.dropout:
            self.linear_1_drop = nn.Dropout(self.dropout[0])
        for i,h in enumerate(self.hidden_layers[1:],1):
            setattr(self, 'linear_'+str(i+1), nn.Linear( self.hidden_layers[i-1], self.hidden_layers[i] ))
            if self.dropout:
                setattr(self, 'linear_'+str(i+1)+'_drop', nn.Dropout(self.dropout[i]))

        # Connect the wide- and dee-side of the model to the output neuron(s)
        self.output = nn.Linear(self.hidden_layers[-1]+self.wide_dim, self.n_class)


    def compile(self, optimizer="Adam", learning_rate=0.001, momentum=0.0):
        
        self.activation, self.criterion = None, F.mse_loss

        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        self.method = 'regression'


    def forward(self, X_w, X_d):

        # Deep Side
        emb = [getattr(self, 'emb_layer_'+col)(X_d[:,self.deep_column_idx[col]].long())
               for col,_,_ in self.embeddings_input]
        if self.continuous_cols:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            cont = [X_d[:, cont_idx].float()]
            deep_inp = torch.cat(emb+cont, 1)
        else:
            deep_inp = torch.cat(emb, 1)

        x_deep = F.relu(self.linear_1(deep_inp))
        if self.dropout:
            x_deep = self.linear_1_drop(x_deep)
        for i in range(1,len(self.hidden_layers)):
            x_deep = F.relu( getattr(self, 'linear_'+str(i+1))(x_deep) )
            if self.dropout:
                x_deep = getattr(self, 'linear_'+str(i+1)+'_drop')(x_deep)

        # Deep + Wide sides
        wide_deep_input = torch.cat([x_deep, X_w.float()], 1)

        if not self.activation:
            out = self.output(wide_deep_input)
        else:
            out = self.activation(self.output(wide_deep_input))

        return out


    def fit(self, dataset, n_epochs, batch_size):

        widedeep_dataset = DatasetLoader(dataset)
        train_loader = torch.utils.data.DataLoader(dataset=widedeep_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        # set the model in training mode
        net = self.train()
        for epoch in range(n_epochs):
            total=0
            correct=0
            for i, (X_wide, X_deep, target) in enumerate(train_loader):
                X_w = Variable(X_wide)
                X_d = Variable(X_deep)
                y = (Variable(target).float() if self.method != 'multiclass' else Variable(target))

                X_w, X_d, y = X_w.to(device), X_d.to(device), y.to(device)

                self.optimizer.zero_grad()
                y_pred =  net(X_w, X_d)
                y_pred = torch.squeeze(y_pred)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()

                if self.method != "regression":
                    total+= y.size(0)
                    if self.method == 'logistic':
                        y_pred_cat = (y_pred > 0.5).squeeze(1).float()
                    if self.method == "multiclass":
                        _, y_pred_cat = torch.max(y_pred, 1)
                    correct+= float((y_pred_cat == y).sum().data[0])
            self.loss_values.append(loss.item())
            print ('Epoch {} of {}, Loss: {}'.format(epoch+1, n_epochs,
                    round(loss.item(),3)))


    def predict(self, dataset):


        X_w = Variable(torch.from_numpy(np.array(dataset.wide, dtype=np.float32))).float()
        X_d = Variable(torch.from_numpy(np.array(dataset.deep, dtype=np.float32)))

        X_w, X_d = X_w.to(device), X_d.to(device)

        # set the model in evaluation mode so dropout is not applied
        net = self.eval()
        pred = net(X_w,X_d).cpu()
        if self.method == "regression":
            return pred.squeeze(1).data.numpy()
        if self.method == "logistic":
            return (pred > 0.5).squeeze(1).data.numpy()
        if self.method == "multiclass":
            _, pred_cat = torch.max(pred, 1)
            return pred_cat.data.numpy()


 

    def get_embeddings(self, col_name):
        params = list(self.named_parameters())
        emb_layers = [p for p in params if 'emb_layer' in p[0]]
        emb_layer  = [layer for layer in emb_layers if col_name in layer[0]][0]
        embeddings = emb_layer[1].cpu().data.numpy()
        col_label_encoding = self.encoding_dict[col_name]
        inv_dict = {v:k for k,v in col_label_encoding.items()}
        embeddings_dict = {}
        for idx,value in inv_dict.items():
            embeddings_dict[value] = embeddings[idx]

        return embeddings_dict

def data_processing_unlabeled(df, wide_cols, embeddings_cols, continuous_cols, scale=False, def_dim=8):
    """
    Processes the input data (unlabeled data) for prediction, creating the necessary wide and deep embeddings.
    """
    if type(embeddings_cols[0]) is tuple:
        emb_dim = dict(embeddings_cols)
        embeddings_cols = [emb[0] for emb in embeddings_cols]
    else:
        emb_dim = {e:def_dim for e in embeddings_cols}
    deep_cols = embeddings_cols + continuous_cols

    # Copy the dataframe so we don't mutate the original one
    df_tmp = df.copy()

    # Extract the categorical column names for one-hot encoding
    categorical_columns = list(df_tmp.select_dtypes(include=['object']).columns)

    # Encoding the categorical columns (if any)
    encoding_dict, df_tmp = encoder(df_tmp)
    encoding_dict = {k: encoding_dict[k] for k in encoding_dict if k in deep_cols}
    embeddings_input = []
    for k, v in encoding_dict.items():
        embeddings_input.append((k, len(v), emb_dim[k]))

    df_deep = df_tmp[deep_cols]
    deep_column_idx = {k: v for v, k in enumerate(df_deep.columns)}

    # Scaling the continuous columns if required
    if scale:
        scaler = StandardScaler()
        for cc in continuous_cols:
            df_deep[cc] = scaler.fit_transform(df_deep[cc].values.reshape(-1, 1))

    df_wide = df_tmp[wide_cols]
    dummy_cols = [c for c in wide_cols if c in categorical_columns]
    df_wide = pd.get_dummies(df_wide, columns=dummy_cols)

    # Prepare the dataset (without labels for unlabeled data)
    dataset = namedtuple('dataset', 'wide, deep, labels')(
        wide=df_wide.values,
        deep=df_deep.values,
        labels=np.zeros(len(df_deep))  # No target labels, so we set this to zero
    )

    # Return all the processed data (wide, deep, embeddings, etc.)
    return {
        'dataset': dataset,
        'embeddings_input': embeddings_input,
        'deep_column_idx': deep_column_idx,
        'encoding_dict': encoding_dict
    }

def recommend_top_k_movies(predict_user, final_df, movie_records, model, wide_cols, embeddings_cols, continuous_cols, search_term = None):

    # Preprocess movie_records data
    movies = movie_records.copy()
    movie_records['genres'] = movie_records.apply(lambda row : row['genres'].split("|")[0],axis=1)
    movie_records['movie_year'] = movie_records.apply(lambda row : int(row['title'].split("(")[-1][:-1]),axis=1)
    movie_records.drop(['title'],axis=1,inplace=True)
    
    # rated and unrated movies by user
    rated_movies = final_df[final_df['userId'] == predict_user]['movieId'].unique()
    unrated_movies = movie_records[~movie_records['movieId'].isin(rated_movies)]
    # user data/info
    user_data = final_df[final_df['userId'] == predict_user].iloc[0]
    user_features = {col: user_data[col] for col in ['userId', 'gender', 'age', 'occupation']}
    # prediction input
    prediction_input = unrated_movies.copy()
    for col, val in user_features.items():
        prediction_input[col] = val
    prediction_df = pd.DataFrame(prediction_input)

    # Process the prediction data using the adjusted data_processing function for unlabeled data
    processed = data_processing_unlabeled(
        prediction_df, 
        wide_cols, 
        embeddings_cols, 
        continuous_cols, 
        scale=True
    )

    # Generate the dataset for prediction and get prediction from model
    dataset = processed['dataset']
    predicted_ratings = model.predict(dataset)

    unrated_movies = np.array(unrated_movies)  # Convert to numpy array if it's a DataFrame or list
    predicted_ratings = np.array(predicted_ratings)
    unrated_movie_ids = unrated_movies[:, 0]  # This gives the movieId (first column)
    
    unrated_movie_ids = np.ravel(unrated_movie_ids)  # Flatten to 1D if necessary
    predicted_ratings = np.ravel(predicted_ratings)  # Flatten to 1D if necessary

    top_k_recommendations = pd.DataFrame({
        'movieId': unrated_movie_ids,
        'predicted_rating': predicted_ratings
    }).sort_values(by='predicted_rating', ascending=False)

    # Convert unrated_movies array into a DataFrame and Merge unrated_movies_df with movie_records to get the movie titles
    unrated_movies_df = pd.DataFrame(unrated_movies, columns=['movieId', 'genres', 'movie_year'])
    unrated_movies_df = unrated_movies_df.merge(movies[['movieId', 'title']], on='movieId', how='left')
    
    # top movies with details
    top_movies_with_details = top_k_recommendations.merge(unrated_movies_df[['movieId', 'title', 'genres', 'movie_year']], 
                                                        on='movieId', how='left')
    
    # If a search term is provided, filter the movies
    if search_term:
        top_movies_with_details = search_query(search_term, top_movies_with_details)

    return top_movies_with_details

# Function to filter based on title and genres
def search_query(input_query, df):
    input_query = input_query.lower()  # Make the input query lowercase to make it case-insensitive

    print(df.columns)
    
    # Filter the rows where the title or genres contain the input_query
    filtered_df = df[df['title'].str.contains(input_query, case=False, na=False) | 
                     df['genres'].str.contains(input_query, case=False, na=False)]
    
    return filtered_df

def save_model(model, filepath):
    """
    Save the model's state dictionary and additional information to a file.
    
    Args:
        model: The instance of the NeuralNet model to save.
        filepath: The path where the model will be saved.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'encoding_dict': model.encoding_dict,
        'loss_values': model.loss_values
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(model_class, model_args, filepath, device):
    """
    Load a model from a saved file.

    Args:
        model_class: The class definition of the model.
        model_args: A dictionary of arguments to initialize the model.
        filepath: The path to the saved model file.
        device: The device on which to load the model (e.g., "cpu" or "cuda").

    Returns:
        An instance of the model with the loaded state and additional data.
    """
    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    
    # Initialize the model
    model = model_class(**model_args)
    
    # Load the state dictionary into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore additional information
    model.encoding_dict = checkpoint.get('encoding_dict', {})
    model.loss_values = checkpoint.get('loss_values', [])
    
    model.to(device)
    print(f"Model loaded from {filepath}")
    return model

# Function to generate and display word cloud
def generate_wordcloud(text):
    # Generate the word cloud with custom settings
    wordcloud = WordCloud(
        background_color='black',
        max_words=200,
        colormap='coolwarm',  # Creative color palette
        contour_width=1,
        contour_color='black',
        random_state=42,
        min_font_size=10,
        max_font_size=200,
        prefer_horizontal=0.5,  # Rotate words
    ).generate(text)
    
    # Display the wordcloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    return plt
