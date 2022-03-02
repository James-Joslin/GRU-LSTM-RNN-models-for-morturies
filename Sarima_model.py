import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from External_Functions import data_Visualisation, Prep_Workspace, get_Data, test_stationarity


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = (15, 6)

Prep_Workspace()
Daily = get_Data()
Daily = Daily[8000:]
data_Visualisation(dataIn=Daily)
Daily.index = Daily["date"]
Daily = Daily.drop('date',axis=1)
print(Daily)
from statsmodels.tsa.stattools import adfuller as adf
test_stationarity(Daily)

output = (adf(Daily['deaths']))
dfoutput = pd.Series(output[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in output[4].items():
        dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

#estimating trend and seasonlity
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(Daily)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(Daily, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

train = Daily[:int(0.8*(len(Daily)))]
valid = Daily[int(0.8*(len(Daily))):]

#plotting the data
ax = train.plot()
valid.plot(ax=ax)
plt.show()

from pmdarima.arima import auto_arima
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True,seasonal=True,m=365,D=1)
model.fit(train)

forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

#plot the predictions for validation set
plt.plot(train, label='Train')
plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.legend(loc='best')
plt.show()

#plot the predictions for validation set
plt.plot(train, label='Train')
plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.legend(loc='best')
plt.show()