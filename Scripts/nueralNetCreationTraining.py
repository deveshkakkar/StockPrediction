import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from sklearn import preprocessing
from yahoo_fin import stock_info as si
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

NueralSteps = 50
LookupStep = 1
Scale = True
scaleStr = f"sc-{int(Scale)}"
Shuffle = True
shuffleStr = f"sh-{int(Shuffle)}"
SplitByDate = False
splitByDateStr = f"sbd-{int(SplitByDate)}"
TestSize = 0.2
MainColumns = ["adjclose", "volume", "open", "high", "low"]
currentDate = time.strftime("%Y-%m-%d")
NueralLayers = 2
CELL = LSTM
UNITS = 256
DROPOUT = 0.4
BIDIRECTIONAL = False
LOSS = "huber_loss"
OPTIMIZER = "adam"
BatchSize = 64
EPOCHS = 150
ticker = "TSLA"
tickerFilename = os.path.join("data", f"{ticker}_{currentDate}.csv")
modelName = f"{currentDate}_{ticker}-{shuffleStr}-{scaleStr}-{splitByDateStr}-\
{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{NueralSteps}-step-{LookupStep}-layers-{NueralLayers}-units-{UNITS}"
if BIDIRECTIONAL:
    modelName += "-b"

if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")

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





data = getData(ticker, NueralSteps, scale=Scale, dateSplit=SplitByDate,
               shuffle=Shuffle, stepLookup=LookupStep, testSize=TestSize,
               mainColumns=MainColumns)
data["df"].to_csv(tickerFilename)
model = modelCreation(NueralSteps, len(MainColumns), loss=LOSS, units=UNITS, cell=CELL, nueralLayers=NueralLayers,
                      dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
checkpointer = ModelCheckpoint(os.path.join("results", modelName + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", modelName))
history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BatchSize,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)

def predict(model, data):
    finalOccurance = data["finalOccurance"][-NueralSteps:]
    finalOccurance = np.expand_dims(finalOccurance, axis=0)
    prediction = model.predict(finalOccurance)
    if Scale:
        predicted_price = data["scaledColumn"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price

def finalDf(model, data):
    profitBuy  = lambda current, trueFuturePrice, predictedFuturePrice: trueFuturePrice - current if predictedFuturePrice > current else 0
    profitSell = lambda current, trueFuturePrice, predictedFuturePrice: current - trueFuturePrice if predictedFuturePrice < current else 0
    Xtest = data["X_test"]
    ytest = data["y_test"]
    ypred = model.predict(Xtest)
    if Scale:
        ytest = np.squeeze(data["scaledColumn"]["adjclose"].inverse_transform(np.expand_dims(ytest, axis=0)))
        ypred = np.squeeze(data["scaledColumn"]["adjclose"].inverse_transform(ypred))
    testDf = data["test_df"]
    testDf[f"adjclose_{LookupStep}"] = ypred
    testDf[f"true_adjclose_{LookupStep}"] = ytest
    testDf.sort_index(inplace=True)
    finalDf = testDf
    finalDf["profitBuy"] = list(map(profitBuy,
                                      finalDf["adjclose"],
                                      finalDf[f"adjclose_{LookupStep}"],
                                      finalDf[f"true_adjclose_{LookupStep}"]))
    finalDf["profitSell"] = list(map(profitSell,
                                       finalDf["adjclose"],
                                       finalDf[f"adjclose_{LookupStep}"],
                                       finalDf[f"true_adjclose_{LookupStep}"]))
    return finalDf

def plotGraph(test):
    plt.plot(test[f'true_adjclose_{LookupStep}'], c='b')
    plt.plot(test[f'adjclose_{LookupStep}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


pathModel = os.path.join("results", modelName) + ".h5"
model.load_weights(pathModel)

loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
if Scale:
    meanAbsoluteError = data["scaledColumn"]["adjclose"].inverse_transform([[mae]])[0][0]
else:
    meanAbsoluteError = mae

futurePrice = predict(model, data)
FinalData = finalDf(model, data)


accuracy = (len(FinalData[FinalData['profitSell'] > 0]) + len(FinalData[FinalData['profitBuy'] > 0])) / len(FinalData)
totalBuyProfit  = FinalData["profitBuy"].sum()
totalSellProfit = FinalData["profitSell"].sum()
totalProfit = totalBuyProfit + totalSellProfit
profitPerTrade = totalProfit / len(FinalData)

print(f"Future price after {LookupStep} days is {futurePrice:.2f}$")
print(f"{LOSS} loss:", loss)
print("Mean Absolute Error:", meanAbsoluteError)
print("Accuracy score:", accuracy)
print("Total buy profit:", totalBuyProfit)
print("Total sell profit:", totalSellProfit)
print("Total profit:", totalProfit)
print("Profit per trade:", profitPerTrade)


plotGraph(FinalData)

print(FinalData.tail(10))
csvResultsFolder = "csv-results"
if not os.path.isdir(csvResultsFolder):
    os.mkdir(csvResultsFolder)
csvFile = os.path.join(csvResultsFolder, modelName + ".csv")
FinalData.to_csv(csvFile)