import os
import pandas as pd
from Scripts.params import *
from tensorflow.keras.layers import LSTM
from Scripts.stockPrediction import modelCreation, getData
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("data"):
    os.mkdir("data")

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