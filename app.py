#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask
from flask import request
import pandas as pd
from prediction_script_test import prediction

app = Flask(__name__)

@app.route('/proses-csv', methods=['POST'])
def proses_csv():    
    file = request.files['csv_file']
    if file:
#        return 'File found'
        # Baca file CSV ke dalam pandas DataFrame
        data = pd.read_csv(file)
        
        # Panggil fungsi prediction dengan data sebagai argumen
        result = prediction(data)
        
        # Ubah hasil prediksi menjadi format yang sesuai (misalnya, JSON)
        json_result = result.to_json(orient='records')
        
        return json_result
    else:
        return 'File not found.'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("5000"), debug=True)

