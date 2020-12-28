import random
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
from sklearn import preprocessing
from yahoo_fin import stock_info as si
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

np.random.seed(827)
tf.random.set_seed(827)
random.seed(827)

def syncShuffle(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def modelCreation(SeqLen, nueralFeatures, units=256, cell=LSTM, nueralLayers=2, dropout=0.3,
                  loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(nueralLayers):
        if i == 0:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, SeqLen, nueralFeatures)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, SeqLen, nueralFeatures)))
        elif i == nueralLayers - 1:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

def getData(ticker, numOfSteps=50, scale=True, shuffle=True, stepLookup=1, dateSplit=True,
            testSize=0.2, mainColumns=['adjclose', 'volume', 'open', 'high', 'low']):
    if isinstance(ticker, str):
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    result = {}
    result['df'] = df.copy()
    for col in mainColumns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if "date" not in df.columns:
        df["date"] = df.index
    if scale:
        scaledColumn = {}
        for column in mainColumns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            scaledColumn[column] = scaler
        result["scaledColumn"] = scaledColumn
    df['future'] = df['adjclose'].shift(-stepLookup)
    finalOccurance = np.array(df[mainColumns].tail(stepLookup))
    df.dropna(inplace=True)
    dataSequence = []
    sequences = deque(maxlen=numOfSteps)
    for entry, target in zip(df[mainColumns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == numOfSteps:
            dataSequence.append([np.array(sequences), target])
    finalOccurance = list([s[:len(mainColumns)] for s in sequences]) + list(finalOccurance)
    finalOccurance = np.array(finalOccurance).astype(np.float32)
    result['finalOccurance'] = finalOccurance
    X, y = [], []
    for seq, target in dataSequence:
        X.append(seq)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    if dateSplit:
        trainingSamples = int((1 - testSize) * len(X))
        result["X_train"] = X[:trainingSamples]
        result["y_train"] = y[:trainingSamples]
        result["X_test"]  = X[trainingSamples:]
        result["y_test"]  = y[trainingSamples:]
        if shuffle:
            syncShuffle(result["X_train"], result["y_train"])
            syncShuffle(result["X_test"], result["y_test"])
    else:
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                    test_size=testSize, shuffle=shuffle)
    dates = result["X_test"][:, -1, -1]
    result["test_df"] = result["df"].loc[dates]
    result["X_train"] = result["X_train"][:, :, :len(mainColumns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(mainColumns)].astype(np.float32)
    return result