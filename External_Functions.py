def clear_directory(directory = ""):
    import os
    for file in os.scandir(directory):
        os.remove(file.path)
    open(directory + "./.keep", 'w').close()

def ExcelToCsv2(excel_in = ".", csv_out = "."):
  import os
  import glob
  import pandas as pd 
  directory = excel_in
  files = glob.glob(os.path.join(directory, "*.xlsx"))
  for a in range(len(files)):
    name = files[a]
    name = name.split("\\")[-1]
    name = name.split(".")[0]
    in_file = pd.read_excel(files[a])
    out_path = csv_out + "/" + name + ".csv"
    in_file.to_csv(out_path, index=False)

def detect_tf_hardware():
    import tensorflow as tf
    from colorama import init, Fore, Back, Style
    init(convert=True)
    GPU = len(tf.config.list_physical_devices('GPU'))
    print(Fore.CYAN + Style.BRIGHT + "Num GPUs Available: " + str(GPU))
    if GPU > 0:
        my_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_visible_devices(devices= my_devices, device_type='GPU')
        print(Fore.CYAN + Style.BRIGHT + "Utilising GPU")
    else:
        my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
        print(Fore.CYAN + Style.BRIGHT + "Could not find a CUDA GPU\nThe programme will use CPU instead")

def yes_or_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        print("Please just use lowercase y/n... Anything else hurts my CPU")
        reply2 = str(input(' (y/n): ')).lower().strip()
        if reply2[0] == 'y':
            print("Thank you")
            return True
        if reply2[0] == 'n':
            print("No worries")
            return False
        else:
            print("Owww!")
            reply3 = str(input(' (y/n): ')).lower().strip()
            if reply3[0] == 'y':
                print("That hurt, but I forgive you")
                return True
            if reply3[0] == 'n':
                print("My brain is a bit sore now, but we got there in the end!")
                return False
            else:
                print("Why would you do this to me? <{;_;}>")
                return yes_or_no(question)

def Prep_Workspace():
    import os
    if os.path.isdir("./model") == True:
        pass
    else:
        os.mkdir("./model")
    if os.path.isdir("./forecasted") == True:
        pass
    else:
        os.mkdir("./forecasted")
    if os.path.isdir("./csvData") == True:
        pass
    else:
        os.mkdir("./csvData")

def sin_transform(values):
    import numpy as np
    return np.sin(2*np.pi*values/len(set(values)))

def cos_transform(values):
    import numpy as np
    return np.cos(2*np.pi*values/len(set(values)))

# Had to combine data sources
def get_Data(cutoff = ""):
    import glob
    import os
    from tqdm import tqdm
    import pandas as pd
    from colorama import Fore, Style
    # check for new files - rebuild csvData directory if there's an inbalance
    shared_data = "Shared Data Location"
    local_data = "./csvData"
    shared_list = glob.glob(os.path.join(shared_data, "*.xlsx"))
    for i in range(len(shared_list)):
        shared_list[i] = shared_list[i].split("\\")[-1].split(".")[0]
    local_list = glob.glob(os.path.join(local_data, "*.csv"))
    for i in range(len(local_list)):
        local_list[i] = local_list[i].split("\\")[-1].split(".")[0]
    new_files = list(set(shared_list) - set(local_list))
    if len(new_files) != 0:
        print("New files found\nRepopulating local data directory")
        clear_directory(directory = local_data)
        ExcelToCsv2(excel_in = shared_data, csv_out = local_data)
    else:
        print("No new files found")

    # concatenate files together
    files = glob.glob(os.path.join(local_data, "*.csv"))
    df_list = []
    print("Concatenating files")
    for a in tqdm(range(len(files))):
        in_file = pd.read_csv(files[a], low_memory=False)
        df_list.append(in_file)
    Results = pd.concat(df_list)
    print(Results.info())

    #  clean data
    merge_date = '2021-11-01'
    Results = Results[Results['DATE_OF_DEATH'].notna()]
    Results['DATE_OF_DEATH'] = pd.to_datetime(Results['DATE_OF_DEATH'], format='%Y%m%d')
    Results = Results.sort_values(by='DATE_OF_DEATH')
    Results = Results[~(Results['DATE_OF_DEATH'] < '1996-01-01')]
    Results = Results[~(Results['DATE_OF_DEATH'] > merge_date)]
    Daily = Results.groupby(['DATE_OF_DEATH']).size().reset_index(name='Deaths')
    Daily = Daily.drop(Daily.tail(1).index)
    Daily.columns = ['date', 'deaths']
    First_Date = str(Daily['date'][0])

    # add registry data
    print("Getting registry data, this can take a while")
    registry = "Registry Location"
    registry_data = glob.glob(os.path.join(registry, "*.xlsx"))
    registry_data = pd.read_excel(registry_data[0], sheet_name="Matrix")
    registry_data = pd.DataFrame(registry_data["DoD"])
    registry_data = registry_data.groupby(['DoD']).size().reset_index(name='Deaths')
    registry_data.columns = ['date', 'deaths']
    registry_data = registry_data[~(registry_data['date'] < merge_date)]
    registry_data = registry_data[~(registry_data['date'] > cutoff)]

    # Combine daily and registry data
    Daily = Daily.append(registry_data)
    print(Fore.CYAN + Style.BRIGHT + "Base Data Details: ")
    print(Daily.head(), Daily.tail(), Daily.info())
    return Daily

def periodicity_Vars(dataIn):
    import pandas as pd
    dataIn['dayofweek'] = dataIn['date'].dt.dayofweek
    dataIn['month'] = dataIn['date'].dt.month
    dataIn['year'] = dataIn['date'].dt.year
    dataIn['day'] = dataIn['date'].dt.day
    dataIn.set_index('date', inplace=True)
    print(dataIn.head())
    return dataIn

def data_Visualisation(dataIn):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.plot(dataIn["date"], dataIn["deaths"])
    plt.show()
    monthly_resample = dataIn
    monthly_resample = monthly_resample.set_index('date')
    monthly_resample = monthly_resample.resample("W").sum()
    monthly_resample['month'] = monthly_resample.index.month
    sns.pointplot(data = monthly_resample, x = 'month', y = 'deaths')
    plt.show()

def model_warnings():
    from colorama import Fore, Style
    print(Fore.CYAN + Style.BRIGHT + "WARNING: If you have changed the forecasting time period, or have added new data, but have an already saved model")
    print(Fore.CYAN + Style.BRIGHT + "then your pre-existing model should be rebuilt if there is significant deviation within the new vs old data.")
    print(Fore.CYAN + Style.BRIGHT + "Furthermore, if this is code newly cloned from github no model will exist as the model files are not pushed to git.")
    print(Fore.CYAN + Style.BRIGHT + "Attempts to access a non-existent model will result in a fatal error.")

def create_uni_dataset(dataset, time_steps = 1):
    import numpy as np
    Xs, ys = [], []
    for i in range(len(dataset)-time_steps-1):
        a = dataset[i:(i+time_steps), 0]
        Xs.append(a)
        ys.append(dataset[i + time_steps, 0])
    return np.array(Xs), np.array(ys)

def split_seq(series, n_past, n_future):
    import numpy as np
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
    # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=12):
        from keras.callbacks import LearningRateScheduler
        import numpy as np
        '''
        Wrapper function to create a LearningRateScheduler with step decay schedule.
        '''
        def schedule(epoch):
            return initial_lr * (decay_factor ** np.floor(epoch/step_size))
        return LearningRateScheduler(schedule)

def load_model(model_name = ""):
    from tensorflow.keras.models import model_from_json
    from colorama import Fore, Style
    import h5py
    print(Fore.CYAN + Style.BRIGHT + "Loading Precomputed Model")
    json_file = open("./model/{}.json".format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./model/{}.h5".format(model_name))
    print(Fore.CYAN + Style.BRIGHT + "Loaded model from disk")
    return model

def save_model(model_In, save_name = ""):
    from colorama import Fore, Style
    import h5py
    # serialize model to JSON
    model_json = model_In.to_json()
    with open("./model/{}.json".format(save_name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model_In.save_weights("./model/{}.h5".format(save_name))
    print(Fore.CYAN + Style.BRIGHT + "Saved model to disk")

def test_stationarity(timeseries):
    import matplotlib.pyplot as plt
    #Determing rolling statistics
    rolmean = timeseries.rolling(7).mean()
    rolstd = timeseries.rolling(7).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    return plt.show()

def plot_verification(
    model, scaler, X_train, X_test,
    daily2_reshape, TIME_STEPS, y_train, y_test,
    timeseries):
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import math
    from colorama import Fore, Style
    import numpy as np

    print(Fore.CYAN + Style.BRIGHT + "Running predictions")
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    print('Train RMSE: ', str(math.sqrt(mean_squared_error(y_train,train_predict))))
    print('Test RMSE: ', str(math.sqrt(mean_squared_error(y_test,test_predict))))

    # shift train predictions for plotting
    trainPredict_plot = np.empty_like(daily2_reshape)
    trainPredict_plot[:, :] = np.nan
    trainPredict_plot[TIME_STEPS:len(train_predict)+TIME_STEPS, :] = train_predict

    # shift test predictions for plotting
    testPredict_plot = np.empty_like(daily2_reshape)
    testPredict_plot[:, :] = np.nan
    testPredict_plot[len(train_predict)+(TIME_STEPS*2)+1:len(daily2_reshape)-1, :] = test_predict
    

    # plot baseline with predictions
    plt.plot(timeseries.index, scaler.inverse_transform(daily2_reshape), 'lightgrey', label = 'Actual')
    rolmean = timeseries.rolling(7).mean()

    print('Test rolling mean RMSE: ', str(math.sqrt(mean_squared_error(rolmean[int(len(rolmean)-len(test_predict)):],test_predict))))

    plt.plot(timeseries.index, rolmean, color='black', label='Rolling Mean')
    plt.plot(timeseries.index, trainPredict_plot, 'g', label = 'Train Predictions')
    plt.plot(timeseries.index, testPredict_plot, 'r', label = 'Test Predictions')

    plt.title("")
    plt.legend()
    plt.show()

def recursive_forecast(n_steps, model, temp_input, x_input):
    from colorama import init, Fore, Style
    import numpy as np
    from tqdm import tqdm
    lst_output = []
    print(Fore.CYAN + Style.BRIGHT + "Forecasting for " + Fore.CYAN + Style.BRIGHT + str(n_steps) + Fore.CYAN + Style.BRIGHT + " days")
    for i in tqdm(range(0, int(n_steps))):
        if(len(temp_input) > n_steps):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
    return lst_output