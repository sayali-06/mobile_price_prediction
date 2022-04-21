import re
from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
from predict import pricepredict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
data_train = pd.read_csv(r'./data.csv')


@app.route('/',methods=['POST','GET'])
def method_name():
    ram = sorted(data_train['RAM'].unique())
    rom = sorted(data_train['ROM'].unique())
    ratings = sorted(data_train['Ratings'].unique())
    mobile_size =sorted(data_train['Mobile_Size'].unique())
    primary_cam = sorted(data_train['Primary_Cam'].unique())
    selfi_cam =sorted(data_train['Selfi_Cam'].unique())
    battery_power = sorted(data_train['Battery_Power'].unique())
    
    return render_template('base.html',ram=ram, rom = rom, ratings=ratings, mobile_size=mobile_size, primary_cam=primary_cam,selfi_cam=selfi_cam, battery_power=battery_power)

@app.route('/predict',methods= ['POST','GET'])
def predict():
    data_train = pd.read_csv(r'./data.csv')
    
    knn = KNeighborsClassifier()
    std = StandardScaler() 

    ram=request.form.get('ram')
    rom=request.form.get('rom')
    ratings=request.form.get('ratings')
    mobile_size=request.form.get('mobile_size')
    primary_cam=request.form.get('primary_cam')
    selfi_cam =request.form.get('selfi_cam')
    battery_power = request.form.get('battery_power')
    
    
    y = data_train['Price']
    
    
    data_train = data_train.drop(['Price'],axis=1)
    data_train = data_train.drop(['Unnamed: 0'],axis=1)
    data_train.drop(['Brand me'],axis=1,inplace=True)
   

    data_train['Ratings'] = data_train['Ratings'].fillna(data_train['Ratings'].mean())
    data_train['RAM'] = data_train['RAM'].fillna(data_train['RAM'].mean())
    data_train['ROM'] = data_train['ROM'].fillna(data_train['ROM'].mean())
    data_train['Mobile_Size'] = data_train['Mobile_Size'].fillna(data_train['Mobile_Size'].mean())
    data_train['Selfi_Cam'] = data_train['Selfi_Cam'].fillna(data_train['Selfi_Cam'].mean())

    
  
    X_std = std.fit_transform(data_train)
    print(X_std)
    print()
    data=np.array([ram, rom, ratings, mobile_size, primary_cam, selfi_cam, battery_power]).reshape(1, 7)
    print(data) 
    knn.fit(data_train,y)
    prediction =knn.predict(pd.DataFrame(columns=['ram', 'rom', 'ratings', 'mobile_size', 'primary_cam','selfi_cam','battery_power'],
                              data=np.array([ram, rom, ratings, mobile_size, primary_cam,selfi_cam, battery_power]).reshape(1, 7)))
    print(prediction)
    return str(np.round(prediction[0],2))


if __name__ == '__main__':
    app.run(debug=True)