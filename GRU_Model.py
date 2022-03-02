# Base Packages
import warnings
# Numpy
import numpy as np
# Pandas
import pandas as pd 
from pandas.plotting import register_matplotlib_converters
# Matplot
from pylab import rcParams
import matplotlib.pyplot as plt
# ML Packages
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Dense, GRU
# Utilities
from External_Functions import data_Visualisation, detect_tf_hardware, get_Data, load_model, model_warnings, plot_verification
from External_Functions import save_model, yes_or_no, Prep_Workspace, create_uni_dataset, step_decay_schedule, recursive_forecast
from colorama import init, Fore, Style

init(autoreset=True)
warnings.simplefilter(action='ignore', category=FutureWarning)
register_matplotlib_converters()
# sns.set(style='whitegrid', palette = 'muted', font_scale = 1.5)
rcParams['figure.figsize'] == 30, 10
detect_tf_hardware()

Random_Seed = 42
np.random.seed(Random_Seed)
tf.random.set_seed(Random_Seed)

# Setup all neccasary directories if missing and collect data
Prep_Workspace()
Daily = get_Data(cutoff="2022-01-10")
data_Visualisation(dataIn=Daily)

# reshape data
daily2_reshape = np.array(Daily.reset_index()['deaths']).reshape(-1,1)
# start making reshaped data suitable for neural network 
scaler = MinMaxScaler()
daily2_reshape = scaler.fit_transform(daily2_reshape)
print("Transformed data for neural network:")
print(daily2_reshape)

# Split into train and test datasets
train_size = int(len(daily2_reshape) * 0.95)
print(train_size)
test_size = len(daily2_reshape) - train_size
train, test = daily2_reshape[0:train_size,:], daily2_reshape[train_size:len(daily2_reshape),:1]
print(Fore.CYAN + Style.BRIGHT + 'Train and Test Shapes:\n', str(train.shape), str(test.shape))

# Create arrays with consideration to time steps
TIME_STEPS = 180
X_train, y_train = create_uni_dataset(train, time_steps = TIME_STEPS)
X_test, y_test = create_uni_dataset(test, time_steps = TIME_STEPS)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Execute pre-existing or build model
model_warnings()
if yes_or_no("Use pre-existing model?") == True:
    # Load Precomputed Model
    model = load_model(model_name="Deaths_6months_GRU_V1")
    print(model.summary())
    temp = Daily
    temp.index = temp["date"]
    temp = temp.drop('date',axis=1)
    # test_stationarity(timeseries=temp)
    plot_verification(
        model = model, scaler=scaler, X_train=X_train, X_test=X_test,
        daily2_reshape = daily2_reshape, TIME_STEPS=TIME_STEPS,
        y_train=y_train, y_test = y_test, timeseries=temp)

else:
    print(Fore.CYAN + Style.BRIGHT + "Building model")
    model = keras.Sequential()
    model.add(
        Bidirectional(
            GRU(
                units = 128,
                input_shape = (X_train.shape[1], X_train.shape[2]),
                return_sequences=True)
            )
        )
    model.add(
        Bidirectional(
            GRU(
                units = 128,
                input_shape = (X_train.shape[1], X_train.shape[2]),
                return_sequences=True)
            )
        )
    model.add(Dropout(rate = 0.2))
    model.add(
        Bidirectional(
            GRU(
                units = 64,
                input_shape = (X_train.shape[1], X_train.shape[2]),
                return_sequences=False)
            )
        )
    model.add(Dropout(rate = 0.2))
    model.add(Dense(8))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')

    lr_sched = step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=8)

    print(Fore.CYAN + Style.BRIGHT + "Fitting model:")
    history = model.fit(
        X_train, y_train,
        epochs = 150,
        batch_size = 128,
        validation_split = 0.2,
        callbacks = [lr_sched],
        shuffle = False
    )
    print(model.summary())
    temp = Daily
    temp.index = temp["date"]
    temp = temp.drop('date',axis=1)
    plot_verification(
        model = model, scaler=scaler, X_train=X_train, X_test=X_test,
        daily2_reshape = daily2_reshape, TIME_STEPS=TIME_STEPS,
        y_train=y_train, y_test = y_test, timeseries=temp)
    save_model(model_In=model, save_name="Deaths_6months_GRU_Example")

#  Forecasting
start_point = test.shape[0] - TIME_STEPS
x_input = test[start_point:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()
output = recursive_forecast(n_steps=TIME_STEPS, model=model, temp_input=temp_input, x_input=x_input)
forecast = np.array(scaler.inverse_transform(output)).reshape(-1)
forecast_dataframe=pd.DataFrame({'GRU':forecast})
forecast_dataframe.to_csv("./forecasted/GRU_forecast.csv", index = False)
plt.plot(forecast, 'r', label = "GRU")
plt.legend()
plt.show()