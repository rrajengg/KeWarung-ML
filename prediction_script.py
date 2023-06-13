#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model


def prediction (input_data):
    model = load_model('model_script_w30.h5')

    # Filter for using column
    using_column=['tgl_order','nama_produk','total','stok']
    input_data=input_data[using_column]
    input_data['tgl_order'] = pd.to_datetime(input_data['tgl_order'])

    #Remove missing value
    input_data=input_data.dropna()

    #Remove unrealistic data (negative value for quantity)
    input_data=input_data.drop(input_data[input_data['total'] < 0].index)
    input_data['date'] = input_data['tgl_order'].dt.date

    unique_item = input_data['nama_produk'].unique()

    quantity = {}
    stock ={}

    product_data = {}
    for i, product in enumerate(unique_item):
        product_data[i] = input_data[input_data['nama_produk'] == product][['tgl_order','total','stok','date']]

    for i in range (len(unique_item)):
        product_data[i].set_index('tgl_order', inplace=True)
        #Grouped data for each day
        quantity[i] = product_data[i]['total'].resample('D').sum()
        quantity[i] = quantity[i].drop(quantity[i][quantity[i] > 1000].index)
        quantity[i] = np.array(quantity[i].values.tolist())

        df_sorted = product_data[i].sort_values('tgl_order', ascending=False)
        df_grouped = df_sorted.groupby('date').head(1)
        df_grouped = df_grouped.sort_values('tgl_order', ascending=True)
        stock[i] = df_grouped['stok']
        stock[i] = np.array(stock[i].values.tolist())

    series = {}
    time = {}

    for i in range (len(quantity)):
        new_data = []
        for index,data in enumerate(quantity[i]):
            new_data.append([index+1,data])

        new_df = pd.DataFrame(new_data, columns = ['No', 'total'])
        time[i] = np.array(new_df['No'].values.tolist())
        series[i] = np.array(new_df['total'].values.tolist())
    
    window_size = 30  # Assuming the same window size as during training

    input_data = {}
    for i in range (len(unique_item)):
        input_data[i] = np.array(series[i][:window_size])  # Take the last 'window_size' elements as 
        input_data[i] = input_data[i].reshape(1, window_size, 1)
    
    # Predict the next 30 values
    forecast_dict = {}
    for i in range (len(unique_item)):
        forecast = []
        for _ in range(30):
            prediction = model.predict(input_data[i])
            forecast.append(int(prediction[0, 0]))
            prediction = np.array([[[prediction[0, 0]]]])
            input_data[i] = np.concatenate((input_data[i][:, 1:], prediction), axis=1)
        forecast_dict[i] = forecast
    
    sum_list = []
    for x in range (len(unique_item)):
        sum = 0
        forecast_result = np.array(forecast_dict[x])
        for i in range (7):
            sum += forecast_result[i]
        sum_list.append(sum)

    prev_week_list = []
    for x in range (len(unique_item)):
        prev_week = 0
        hari_mulai = window_size - 7
        data_series = np.array(series[x])
        for i in range (hari_mulai, window_size):
            prev_week += data_series[i]
        prev_week_list.append(prev_week)
    
    df = pd.DataFrame(columns=['jenis barang', 'penjualan minggu ini', 'jumlah stok saat ini', 'prediksi penjualan minggu depan', 'estimasi jumlah yang harus di restok'])

    for i  in range (len(unique_item)):
        if sum_list[i] >= stock[i][window_size]:
            total_restock = sum_list[i]-stock[i][window_size]
            new_row = pd.DataFrame([[unique_item[i], prev_week_list[i], stock[i][window_size], sum_list[i], total_restock]], columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
            df = df.sort_values('estimasi jumlah yang harus di restok', ascending=True)

    return df     

