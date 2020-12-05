import os.path
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

BASE_PATH="c:/tmp/yf/"
ClassIdUp = 1
ClassIdDown = 2


#== Hier wird definiert, welche Aktien benutzt werden
#TICKERS = [ "MSFT", "BAYN.DE", "^DJI"]
#TICKERS = ["ATVI"]
TICKERS=[
"ATVI"
,"ADBE"
,"AMD"
,"ALXN"
,"ALGN"
,"GOOG"
,"GOOGL"
,"AMZN"
,"AMGN"
,"ADI"
    ]
"""
,"ANSS"
,"AAPL"
,"AMAT"
,"ASML"
,"ADSK"
,"ADP"
,"BIDU"
,"BIIB"
,"BMRN"
,"BKNG"
,"AVGO"
,"CDNS"
,"CDW"
,"CERN"
,"CHTR"
,"CHKP"
,"CTAS"
,"CSCO"
,"CTXS"
,"CTSH"
,"CMCSA"
,"CPRT"
,"COST"
,"CSX"
    ]

,"DXCM"
,"DOCU"
,"DLTR"
,"EBAY"
,"EA"
,"EXC"
,"EXPE"
,"FB"
,"FAST"
,"FISV"
,"FOX"
,"FOXA"
,"GILD"
,"IDXX"
,"ILMN"
,"INCY"
,"INTC"
,"INTU"
,"ISRG"
,"JD"
,"KLAC"
,"LRCX"
,"LBTYA"
,"LBTYK"
,"LULU"
,"MAR"
,"MXIM"
,"MELI"
,"MCHP"
,"MU"
,"MSFT"
,"MRNA"
,"MDLZ"
,"MNST"
,"NTES"
,"NFLX"
,"NVDA"
,"NXPI"
,"ORLY"
,"PCAR"
,"PAYX"
,"PYPL"
,"PEP"
,"PDD"
,"QCOM"
,"REGN"
,"ROST"
,"SGEN"
,"SIRI"
,"SWKS"
,"SPLK"
,"SBUX"
,"SNPS"
,"TMUS"
,"TTWO"
,"TSLA"
,"TXN"
,"KHC"
,"TCOM"
,"ULTA"
,"VRSN"
,"VRSK"
,"VRTX"
,"WBA"
,"WDC"
,"WDAY"
,"XEL"
,"XLNX"
,"ZM"
]
"""
MODEL_ROW_TICKS = 14

def update_stocks( save_path, tickers ):
    for t in tickers:
        #stock_data = yf.Ticker(t)
        #hist = stock_data.history(period="max", interval="1h")

        if not os.path.isfile("%s/%s.csv" % (save_path, t)):
            hist = yf.download(t,
                        start='2019-01-01',
                        end='2020-10-13',
                        progress=True,
                        interval="1h")

            hist.to_csv("%s/%s.csv" % (save_path,t))


def load_stocks( load_path, tickers ):
    stock_map = {}
    for t in tickers:

        df = pd.read_csv("%s/%s.csv" % (load_path,t))
        stock_map[t] = df
        #print(df)

    return stock_map


""" 
Analises one single stock data. 
params:
   hist - pandas data frame with stock data
"""
def analyse_one(hist):

    cnt = 0
    model_data = []
    response_data = []

    last_row = []
    last_bo_row = []

    np_data = np.c_[
        hist["Open"].to_numpy(),
        hist["Close"].to_numpy(),
        hist["High"].to_numpy(),
        hist["Low"].to_numpy(),
        #hist["Volume"].to_numpy(),
    ]

    np_data = normalize(np_data, axis=0, norm='max')

    print("Kurse-Werte Matrix Dimensionen und Auszug:", np_data.shape )
    print(np_data)
    UsedColumns = np_data.shape[1]

    for d in range(0,len(np_data)-(MODEL_ROW_TICKS)):  #Fuer jede Zeile (Tick - Tag, Stunde, etc)
        model_row = []
        cnt=0
        for row in np_data[d:d+MODEL_ROW_TICKS]:     #Ab der aktueller Zeile N-Ticks fuer ML-Data Zeile
            model_row.extend(row)
            print(cnt, " : ", model_row)


        if len(model_data) == 0:    #Wenn Modell leer ->
            model_data = model_row  # erste Zeile im Modell
        else:
            model_data = np.vstack([model_data, model_row]) # eine neue Zeile hinzufuegen

        idxLast = MODEL_ROW_TICKS*UsedColumns-1
        idxLastButOne = (MODEL_ROW_TICKS-1)*UsedColumns-1
        valueLast =   model_row[idxLast]
        valueLastButOne = model_row[idxLastButOne]

        resultValue =  valueLast - valueLastButOne

        if ( resultValue < 0 ):
            response_data.append(ClassIdDown)
            print ("Down")
        else:
            response_data.append(ClassIdUp)
            print("Up")

        pass

    X = np.array(model_data)[:, 0:5 * (MODEL_ROW_TICKS - 1)]
    Xunscaled = np.array(model_data)[:, 0:5*(MODEL_ROW_TICKS-1)]

    #X = Xunscaled / Xunscaled.max(axis=0)
    y= np.array(response_data)

    print(len(X))
    print(len(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    print(X)
    print(X_train)

    #== Hier Algorithmus auswaehlen
    #clf = svm.SVC(kernel='rbf', C=1)
    clf = AdaBoostClassifier()
    #clf = MLPClassifier(alpha=1, max_iter=1000)
    #clf = DecisionTreeClassifier(max_depth=5)
    #clf = GaussianNB()
    #clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

    #== Hier wird gelernt
    clf.fit(X_train, y_train)

    #== hier wird getestet
    scr = clf.score(X_test, y_test)
    return scr


def get_data_as_np(hist):
    model_data = []
    np_data = np.c_[
        hist["Open"].to_numpy(),
        hist["Close"].to_numpy(),
        hist["High"].to_numpy(),
        hist["Low"].to_numpy(),
        #hist["Volume"].to_numpy(),
    ]

    UsedColumns = np_data.shape[1]
    for d in range(0,len(np_data)-(MODEL_ROW_TICKS)):  #Fuer jede Zeile (Tick - Tag, Stunde, etc)
        model_row = []
        cnt=0
        for row in np_data[d:d+MODEL_ROW_TICKS]:     #Ab der aktueller Zeile N-Ticks fuer ML-Data Zeile
            model_row.extend(row)
            #print(cnt, " : ", model_row)


        if len(model_data) == 0:    #Wenn Modell leer ->
            model_data = model_row  # erste Zeile im Modell
        else:
            model_data = np.vstack([model_data, model_row]) # eine neue Zeile hinzufuegen


    return model_data


def get_response_of(hist):
    model_data = []
    response_data = []

    np_data = np.c_[
        #hist["Open"].to_numpy(),
        hist["Close"].to_numpy(),
        #hist["High"].to_numpy(),
        #hist["Low"].to_numpy(),
        #hist["Volume"].to_numpy(),
    ]

    UsedColumns = np_data.shape[1]

    for d in range(0,len(np_data)-(MODEL_ROW_TICKS)):  #Fuer jede Zeile (Tick - Tag, Stunde, etc)
        model_row = []
        for row in np_data[d:d+MODEL_ROW_TICKS]:     #Ab der aktueller Zeile N-Ticks fuer ML-Data Zeile
            model_row.extend(row)

        idxLast = MODEL_ROW_TICKS*UsedColumns-1
        idxLastButOne = (MODEL_ROW_TICKS-1)*UsedColumns-1
        valueLast =   model_row[idxLast]
        valueLastButOne = model_row[idxLastButOne]

        resultValue =  valueLast - valueLastButOne

        if ( resultValue < 0 ):
            response_data.append(ClassIdDown)
            #print ("Down")
        else:
            response_data.append(ClassIdUp)
            #print("Up")


    y= np.array(response_data)

    return y



if __name__ == '__main__':

    #== Aktiviere untere Zeile, wenn die Aktien noch nicht geladen sind
    update_stocks(BASE_PATH, TICKERS)

    #== Geladenen kurse von der lokaler Platte laden
    stocks_map = load_stocks(BASE_PATH, TICKERS)

    #== Analyse starten fuer eine Aktie
    ticker = "ADI"
    #scr = analyse_one(stocks_map[ticker])
    #print(ticker, ":", scr)

    y = get_response_of(stocks_map[ticker])
    print(y.shape)

    model_data  = []

    for t in TICKERS:
        d = get_data_as_np( stocks_map[t] )
        print("appending %s DIM:%d,%d" % (t, d.shape[0],d.shape[1]))


        if len(model_data) == 0:  # Wenn Modell leer ->
            model_data = d  # erste Zeile im Modell
        else:
            if model_data.shape[0] == d.shape[0]:
                model_data = np.hstack([model_data, d])
            else:
                print("Dimension mismatch, %s skipped" % (t))

    print(model_data.shape)

    model_data = normalize(model_data, axis=0, norm='max')

    X_train, X_test, y_train, y_test = train_test_split(model_data, y, test_size=0.25)

    # == Hier Algorithmus auswaehlen
    Classifiers = [
        #svm.SVC(kernel='rbf', C=1)
        #,svm.SVC(kernel='linear', C=1)
        svm.SVC(kernel='linear', C=10)
        #,svm.SVC(kernel='poly', C=10)
        #, svm.SVC(kernel='sigmoid', C=10)
        ##,AdaBoostClassifier()
        #,MLPClassifier(alpha=1, max_iter=1000)
        ,DecisionTreeClassifier(max_depth=15)
        ,GaussianNB()
        ,RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        , RandomForestClassifier(max_depth=5, n_estimators=10, max_features=2)
        , RandomForestClassifier(max_depth=5, n_estimators=10, max_features=5)
        , RandomForestClassifier(max_depth=5, n_estimators=10, max_features=10)
        , RandomForestClassifier(max_depth=7, n_estimators=10, max_features=1)
        , RandomForestClassifier(max_depth=10, n_estimators=10, max_features=2)
        , RandomForestClassifier(max_depth=5, n_estimators=100, max_features=5)

    ]

    # == Hier wird gelernt
    for clf in Classifiers:
        clf.fit(X_train, y_train)
        scr = clf.score(X_test, y_test)
        print(scr)

