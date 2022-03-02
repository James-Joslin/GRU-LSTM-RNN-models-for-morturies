# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import nan, polyfit, poly1d

# Initial Dataset
init_df = pd.read_csv("./Matrix_Temp.csv")
init_df['DoD'] = pd.to_datetime(init_df['DoD'], dayfirst=True)
init_df = init_df.sort_values(['DoD']).reset_index(drop = True)
init_df = init_df[~(init_df['DoD'] < '2020-03-02')]

init_df['PoD'] = init_df['PoD'].str.title()

Weekly_Index = init_df
print(Weekly_Index["PoD"].unique())
print(Weekly_Index.head())

Weekly_Index = pd.DataFrame(Weekly_Index.groupby(["DoD"]).size()).reset_index()
Weekly_Index = Weekly_Index[~(Weekly_Index['DoD'] < '2020-03-01')]
Weekly_Index = Weekly_Index[~(Weekly_Index['DoD'] > '2021-11-07')]
Weekly_Index.index = Weekly_Index["DoD"]
Weekly_Index = Weekly_Index.drop(["DoD"], axis = 1)
Weekly_Index.rename({0 : 'Count'}, axis = 1, inplace = True)
Weekly_Index = Weekly_Index.resample("W").sum()
print(Weekly_Index)

# Moving average
Weekly_Index['mov_avg'] = Weekly_Index['Count'].rolling(4).mean()


# Get Past Forecasts
past_forecasts = pd.read_csv() # Location of forecasts
print(past_forecasts.head(10))
past_forecasts["date"] =  pd.to_datetime(past_forecasts['date'], dayfirst=True)
past_forecasts.to_csv() # Output for forecasts to shared location - not on local machine
past_forecasts.index = past_forecasts["date"]
past_forecasts = past_forecasts.drop(["date"], axis = 1)
print(past_forecasts.info())

# Calculate deviation
dev_weekly = Weekly_Index[~(Weekly_Index.index < '2021-04-04')]
dev_forecasts = past_forecasts[~(past_forecasts.index > '2021-11-07')]
dev_series = pd.concat([dev_weekly['Count'], dev_forecasts['forecasted deaths']])
print(dev_series)
print(dev_series.std()/7)

# Plot Weekly
plt.plot(Weekly_Index.index, Weekly_Index["Count"], label = "Actual")
plt.plot(Weekly_Index.index, Weekly_Index["mov_avg"], label = "Actual - Moving Average", linestyle = "--")

# Plot forecasts
plt.plot(past_forecasts.index, past_forecasts["forecasted deaths"], label = "Forecasted")
# plt.plot(past_forecasts.index, past_forecasts["low CI"], linestyle = "dashed", label = "Lower Prediction Interval")
plt.plot(past_forecasts.index, past_forecasts["high CI"], label = "Higher Prediction Interval")

# Plot Forecast point line plus note
plt.axvline(pd.to_datetime('2021-04-05'), color='black', linestyle='--', lw=1)
plt.annotate('Forecast Start Point', xy = (pd.to_datetime('2021-04-08'),260))

# Present Plot
plt.xlabel("Date")
plt.ylabel("New Deaths/Week")
plt.legend()
plt.show()


