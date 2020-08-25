import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from sklearn import preprocessing
from collections import deque
import random
import yfinance as yf
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


"""
NAS.OL      Norwegian Air Shuttle ASA   NOK
RYAA        Ryanair Holdings plc        USD
LHA.DE      Deutsche Lufthansa AG       EUR
IAG.MC      International Consolidated Airlines Group   EUR
AF.PA       Air France-KLM SA           EUR
EZJ.L       easyJet plc                 GBp

"""

ticker = ['NAS.OL', 'RYAAY', 'LHA.DE', 'AF.PA', 'EZJ.L']
names = ['Norwegian Air Shuttle ASA', 'Ryanair Holdings plc', 'Deutsche Lufthansa AG'
        'Air France-KLM SA',
         'easyJet plc']



def runPrediction(days_in_past):

    def preprocess_df(df):
        for col in df.columns:
            if col != "target":
                df[col] = df[col].pct_change() # normalize after change
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)
                df[col] = preprocessing.scale(df[col].values)
        df.dropna(inplace=True)

        sequential_data = []
        prev_days = deque(maxlen=SEQ_LEN)

        print(df[-(100 + DAYS_IN_PAST):])
        for i in df.values[-(100 + DAYS_IN_PAST):]:
            prev_days.append([n for n in i[:-1]])
            if len(prev_days) == SEQ_LEN:
                sequential_data.append([np.array(prev_days), i[-1]])
                break

        X = []
        y = []

        for seq, target in sequential_data:
            X.append(seq)
            y.append(target)

        return np.array(X)

    SEQ_LEN = 100
    FUTURE_PERIOD_PREDICT = 7  
    TICKER_TO_PREDICT = "EZJ.L"
    EPOCHS = 10
    BATCH_SIZE = 100
    DAYS_IN_PAST = days_in_past
    NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    # download data
    main_df = pd.DataFrame()

    print()
    print("Downloading newest data ...")
    for i in range(0,len(ticker)):
        df = yf.download(tickers = ticker[i], period = "max", interval = "1d")
        df.rename(columns={"Adj Close":f"{ticker[i]}_close", "Volume": f"{ticker[i]}_volume"}, inplace=True)
        df = df[[f"{ticker[i]}_close", f"{ticker[i]}_volume"]]

        if len(main_df) == 0:
            main_df = df
        else:
            main_df = main_df.join(df)

                                        # modell navn
    model = tf.keras.models.load_model('models/100-SEQ-2-PRED-1595440173')

    main_df['target'] = 1
    pred_x = preprocess_df(main_df)
    prediction = model.predict(pred_x)

    up   = round(prediction[0][0]*100, 2)
    down = round(prediction[0][1]*100, 2)
    change = round(main_df["NAS.OL_close"].iloc[-(1+DAYS_IN_PAST)],2)
    #print()
    print("days_in_past, ticker, daily_change, up_pct, down_pct")
    print(f"{DAYS_IN_PAST}, {TICKER_TO_PREDICT}, {change}%, {up}%, {down}%")
    return DAYS_IN_PAST, TICKER_TO_PREDICT, change, up, down

runPrediction(0) # antall dager tilbake, 0 = idag
