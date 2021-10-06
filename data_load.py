from layers.utils import text_preprocessing, load_word_embedding_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import utils

def get_data_set():
    folder_path =  '../capsule_network_files/'

    lankadeepa_data_path = folder_path + 'corpus/lankadeepa_tagged_comments.csv'
    gossip_lanka_data_path = folder_path + 'corpus/gossip_lanka_tagged_comments.csv'

    EMBEDDING_SIZE = 300
    embedding_type = "fasttext"
    context = 5
    embeds = "fasttext"

    word_embedding_matrix_path = './embeddings/fasttext_lankadeepa_gossiplanka_300_5'

    lankadeepa_data = pd.read_csv(lankadeepa_data_path)[:9059]
    gossipLanka_data = pd.read_csv(gossip_lanka_data_path)
    gossipLanka_data = gossipLanka_data.drop(columns=['Unnamed: 3'])

    all_data = pd.concat([lankadeepa_data, gossipLanka_data], ignore_index=True)
    all_data['label'] = all_data['label'] - 2

    comments_text, labels = text_preprocessing(all_data)
    t = Tokenizer()
    t.fit_on_texts(comments_text)
    vocab_size = len(t.word_index) + 1
    print(vocab_size)

    encoded_docs = t.texts_to_sequences(comments_text)
    max_length = 30
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    comment_labels = np.array(labels)
    comment_labels = utils.to_categorical(comment_labels)
    padded_docs = np.array(padded_docs)

    print("Shape of all comments: ", padded_docs.shape)
    print("Shape of labels: ", comment_labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(padded_docs, comment_labels, test_size=0.1, random_state=42,
                                                        shuffle=True)
    print("\nTrain set shape: ", X_train.shape)
    print("Train label set shape: ", y_train.shape)
    print("Test set shape: ", X_test.shape)
    print("Test label set shape: ", y_train.shape)
    print("\n")
    return X_train,y_train, X_test,y_test

