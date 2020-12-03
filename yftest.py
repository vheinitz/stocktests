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
TICKERS = ["ATVI"]
dummy=["ADBE"
,"AMD"
,"ALXN"
,"ALGN"
,"GOOG"
,"GOOGL"
,"AMZN"
,"AMGN"
,"ADI"
,"ANSS"
,"AAPL"]
MODEL_ROW_TICKS = 10

def update_stocks( save_path, tickers ):
    for t in tickers:
        #stock_data = yf.Ticker(t)
        #hist = stock_data.history(period="max", interval="1h")

        hist = yf.download(t,
                    start='2019-01-01',
                    end='2020-10-13',
                    progress=True,
                    interval="1D")

        hist.to_csv("%s/%s.csv" % (save_path,t))


def load_stocks( load_path, tickers ):
    stock_map = {}
    for t in tickers:

        df = pd.read_csv("%s/%s.csv" % (load_path,t))
        print(df)
        #df = df.resample('d', on='Date').mean().dropna(how='all')
        stock_map[t] = df
        print(df)

    return stock_map


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



if __name__ == '__main__':

    #== Aktiviere untere Zeile, wenn die Aktien noch nicht geladen sind
    #update_stocks(BASE_PATH, TICKERS)

    #== Geladenen kurse von der lokaler Platte laden
    stocks_map = load_stocks(BASE_PATH, TICKERS)

    #== Analyse starten fuer eine Aktie
    ticker = "ATVI"
    scr = analyse_one(stocks_map[ticker])

    print(ticker, ":", scr)


"""
def analyse_one( stocks_map, use_ticker ):
    model_row = []
    cnt = 0
    model_data = []
    response_data = []

    last_row = []
    last_bo_row = []

    np_data=[]
    for ticker in stocks_map:
        hist = stocks_map[ticker]
        tmp = np.c_[
            hist["Open"].to_numpy(),
            hist["Close"].to_numpy(),
            hist["High"].to_numpy(),
            hist["Low"].to_numpy(),
            hist["Volume"].to_numpy(),
        ]

        rows = len(tmp)
        #tmp = tmp[int(rows*2/3): rows, :]
        #tmp = tmp[int(rows/2): rows, :]
        if len(np_data) == 0:
            np_data = tmp
        else:
            #print(np_data.shape)
            #print(tmp.shape)
            np_data = np.hstack([np_data, tmp])

    rows = len(np_data)
    np_data = normalize(np_data, axis=0, norm='max')

    #print (np_data.shape)
    #print(np_data)

    for d in range(0,len(np_data)-MODEL_ROW_DAYS):
        for row in np_data[d:d+MODEL_ROW_DAYS]:
            model_row.extend(row)
            print(cnt, " : ", model_row)
            cnt += 1
            if cnt>2:
                last_bo_row = last_row
            last_row = row

        if len(model_data) == 0:
            model_data = model_row
        else:
            model_data = np.vstack([model_data, model_row])

        #if ticker == use_ticker:
        res = np.subtract( last_bo_row, last_row )
        response_row=[]
        for r in res:
            if ( r<0):
                response_row.append(1)
            else:
                response_row.append(2)
        response_data.append( response_row )
        model_row=[]
        cnt=0

    X = np.array(model_data)[:, 0:5*(MODEL_ROW_DAYS-1)] #[::5*(MODEL_ROW_DAYS-1)]
    y= np.array(response_data)[:,4]

    print(len(X))
    print(len(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    #== Hier Algorithmus auswaehlen
    #clf = svm.SVC(kernel='rbf', C=1)
    #clf = AdaBoostClassifier()
    #clf = MLPClassifier(alpha=1, max_iter=1000)
    #clf = DecisionTreeClassifier(max_depth=5)
    #clf = GaussianNB()
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

    #== Hier wird gelernt
    clf.fit(X_train, y_train)

    #== hier wird getestet
    scr = clf.score(X_test, y_test)
    print(use_ticker, ":", scr)
"""