import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools import indicators

def renkoBetween( df, start_date, end_date ):
    # greater than the start date and smaller than the end date
    mask = (df['date'] > start_date) & (df['date'] <= end_date)
    print(df.loc[mask])

df = pd.read_csv('c:/tmp/yf/ADI.csv')
df.columns = [i.lower() for i in df.columns]

renko = indicators.Renko(df)
print('\n\nRenko box calcuation based on periodic close')
renko.brick_size = 3
renko.chart_type = indicators.Renko.PERIOD_CLOSE
data = renko.get_ohlc_data()
print(data.tail(50))

renkoBetween( data, "2020-05-01", "2020-07-01" )

#data.plot()
#plt.show()


