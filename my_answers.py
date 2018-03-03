import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
import re

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    for v in range(len(series) - window_size):
        X.append(series[v:v+window_size])
        y.append(series[v+window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1, activation='tanh'))
    return model
    

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    
    # set(text) = {';', '@', '(', '/', "'", '1', ' ', '!', '?', 'é', '3', '"', '9', 'g', '&', 'w', 'c', 'j', 'l', 'v', 'e', 'a', 'z', '4', 'h', 'à', 'b', '8', 'è', '0', ',', '2', 'i', 'd', 'y', 'k', 'r', 'm', 'o', '5', '.', '%', 's', '$', ')', 'â', 'f', 'n', '*', 'u', '7', 'p', 'x', '-', 'q', ':', '6', 't'}
    
    text = text.lower()
    text = re.sub(r'[^a-z!,.:;?]', ' ', text)
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    for v in range(0, len(text) - window_size, step_size):
        inputs.append(text[v:v+window_size])
        outputs.append(text[v+window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
