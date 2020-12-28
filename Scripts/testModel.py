import numpy as np
from params import *
import matplotlib.pyplot as plt
from stockPrediction import getData, modelCreation

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