import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pickle import load
from sklearn.metrics import mean_squared_error


########### Test dataset #########

# Load scaler/ index
X_scaler = load(open('X_scaler.pkl', 'rb'))
y_scaler = load(open('y_scaler.pkl', 'rb'))
train_predict_index = np.load("index_train.npy", allow_pickle=True)
test_predict_index = np.load("index_test.npy", allow_pickle=True)

# Load test dataset/ model
G_model = tf.keras.models.load_model('WGAN_google.keras')
# G_model = tf.keras.models.load_model('gen_model_3_1_164.keras')
X_test = np.load("X_test.npy", allow_pickle=True)
y_test = np.load("y_test.npy", allow_pickle=True)
X_train = np.load("X_train.npy", allow_pickle=True)
y_train = np.load("y_train.npy", allow_pickle=True)

def get_test_plot(X_test, y_test, X_train, y_train):
    # Set output steps
    output_dim = y_test.shape[1]

    # Get predicted data
    y_predicted = G_model(X_test)
    y_train_predicted = G_model(X_train)
    rescaled_real_y = y_scaler.inverse_transform(y_test)
    rescaled_predicted_y = y_scaler.inverse_transform(y_predicted)

    rescaled_real_y_train = y_scaler.inverse_transform(y_train)
    rescaled_predicted_y_train = y_scaler.inverse_transform(y_train_predicted)
    print(rescaled_real_y_train.shape)
    print(rescaled_real_y)
    # rescaled_predicted_y = np.concatenate((rescaled_predicted_y_train, rescaled_predicted_y), axis=0)
    # rescaled_real_y = np.concatenate((rescaled_real_y_train, rescaled_real_y), axis=0)
    ## Predicted price
    predict_result = pd.DataFrame()
    for i in range(rescaled_predicted_y.shape[0]):
        y_predict = pd.DataFrame(rescaled_predicted_y[i], columns=["predicted_price"],
                                 index=test_predict_index[i:i + output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

    ## Real price
    real_price = pd.DataFrame()
    for i in range(rescaled_real_y.shape[0]):
        y_train = pd.DataFrame(rescaled_real_y[i], columns=["real_price"], index=test_predict_index[i:i + output_dim])
        real_price = pd.concat([real_price, y_train], axis=1, sort=False)

    predict_result['predicted_mean'] = predict_result.mean(axis=1)
    real_price['real_mean'] = real_price.mean(axis=1)

    predict_result_train = pd.DataFrame()
    for i in range(rescaled_predicted_y_train.shape[0]):
        y_predict = pd.DataFrame(rescaled_predicted_y_train[i], columns=["predicted_price"], index=train_predict_index[i:i+output_dim])
        predict_result_train = pd.concat([predict_result_train, y_predict], axis=1, sort=False)

    real_price_train = pd.DataFrame()
    for i in range(rescaled_real_y_train.shape[0]):
        y_train = pd.DataFrame(rescaled_real_y_train[i], columns=["real_price"], index=train_predict_index[i:i+output_dim])
        real_price_train = pd.concat([real_price_train, y_train], axis=1, sort=False)

    predict_result_train['predicted_mean'] = predict_result_train.mean(axis=1)
    real_price_train['real_mean'] = real_price_train.mean(axis=1)

    predict_final = pd.concat([predict_result_train['predicted_mean'], predict_result['predicted_mean']], axis=0)
    real_final = pd.concat([real_price_train['real_mean'], real_price['real_mean']], axis=0)
    #drop 2020
    # Input_Before = '2020-01-01'
    # predict_result = predict_result.loc[predict_result.index < Input_Before]
    # real_price = real_price.loc[real_price.index < Input_Before]

    # Plot the predicted result
    plt.figure(figsize=(16, 8))
    plt.plot(real_final)
    plt.plot(predict_final, color='r')
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
    plt.title("The result of test", fontsize=20)
    plt.show()
    plt.savefig('test_plot.png')
    # Calculate RMSE
    predicted = predict_result["predicted_mean"]
    real = real_price["real_mean"]
    For_MSE = pd.concat([predicted, real], axis=1)
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    print('-- RMSE -- ', RMSE)

    MAPE = np.mean(np.abs((real - predicted) / real)) * 100
    print('-- MAPE -- ', MAPE)
    return predict_result, RMSE


test_predicted, test_RMSE = get_test_plot(X_test, y_test, X_train, y_train)
test_predicted.to_csv("test_predicted.csv")

# ######### Test dataset #########
# ##### For last set #########
# # Rescale back the real dataset
#
# X_scaler = load(open('X_scaler.pkl', 'rb'))
# y_scaler = load(open('y_scaler.pkl', 'rb'))
# train_predict_index = np.load("index_train.npy", allow_pickle=True)
# test_predict_index = np.load("index_test.npy", allow_pickle=True)
#
# # Load model
# G_model = tf.keras.models.load_model('gen_GRU_model_89.h5')
#
# X_test = np.load("X_test.npy", allow_pickle=True)
# y_test = np.load("y_test.npy", allow_pickle=True)
#
# y_test_hat = G_model(X_test[-1].reshape(1, X_test[-1].shape[0], X_test[-1].shape[1]))
# rescaled_real_ytest = y_scaler.inverse_transform(y_test[-32:])
# rescaled_predicted_ytest = y_scaler.inverse_transform(y_test_hat)
# output_dim = 3
#
# ## Real price
# real_price = pd.DataFrame()
# for i in range(rescaled_real_ytest.shape[0]):
#     test_predict_index = test_predict_index[-34:]
#     y_train = pd.DataFrame(rescaled_real_ytest[i], columns=["real_price"], index=test_predict_index[i:i+output_dim])
#     real_price = pd.concat([real_price, y_train], axis=1, sort=False)
#
# ## Predicted price
# predict_result = pd.DataFrame()
# y_predict = pd.DataFrame(rescaled_predicted_ytest[0], columns=["predicted_price"], index=test_predict_index[-3:])
# predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)
#
#
#
# predict_result['predicted_mean'] = predict_result.mean(axis=1)
# real_price['real_mean'] = real_price.mean(axis=1)
# #
# # Plot the predicted result
# plt.figure(figsize=(16, 8))
# plt.plot(real_price["real_mean"])
# plt.plot(predict_result["predicted_mean"], color = 'r')
# plt.xlabel("Date")
# plt.ylabel("Stock price")
# plt.ylim(0, 100)
# plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
# plt.title("The result of the last set of testdata", fontsize=20)
# plt.show()
# plt.savefig('single_test_plot.png')
#
# # Calculate RMSE
# predicted = predict_result["predicted_mean"]
# real = real_price["real_mean"]
# For_MSE = pd.concat([predicted, real], axis = 1)
# RMSE = np.sqrt(mean_squared_error(predicted, real[-3:]))
# print('-- test dataset RMSE -- ', RMSE)
