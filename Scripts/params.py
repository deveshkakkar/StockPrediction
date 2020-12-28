import os
import time
from tensorflow.keras.layers import LSTM

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
EPOCHS = 5
ticker = "TSLA"
tickerFilename = os.path.join("data", f"{ticker}_{currentDate}.csv")
modelName = f"{currentDate}_{ticker}-{shuffleStr}-{scaleStr}-{splitByDateStr}-\
{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{NueralSteps}-step-{LookupStep}-layers-{NueralLayers}-units-{UNITS}"
if BIDIRECTIONAL:
    modelName += "-b"