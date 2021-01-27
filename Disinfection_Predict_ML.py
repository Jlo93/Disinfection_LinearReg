#Model Used is a multivariate linear regression model to predict 
#Model created and put working on 30/10/20 model trained on Swanlinbar historical data
#Model shows good accuracy in early tests but could be improved a lot, training data from 
#Swanlinbar was of poor quality & not a true representation of plant operation as the 
#plant was in a phase of commissioning & had only been collecting data for a week before 
#data collection. With a bigger quantity of data along with better quality plant operation   
#this models accuary would be greatly improved. 


#Importing Libraries

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
import matplotlib.pyplot as plt
import struct


from mpl_toolkits.mplot3d import Axes3D
from sklearn import neighbors,datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
#from pyModbus.constants import Endian
#from pymodbus.payload import BinaryPayloadDecoder
#from pymodbus.payload import BinaryPayloadBuilder
#from pymodbus.client.sync import ModbusTcpClient as client

#Defines the client PLC's IP address for communications
#client = ModbusTcpClient('192.168.120.11')
#Establishes connection and checks comms status, returns true if comms = OK
#client = connect()
#function reads the %MW5524 location from the PLC's memory this will be repeated for each tag.
#rr = client.read_holdiing_registers(5524,2)
#Sets up the decoder to decode the chosen memory word, endian is set to big as the M340 byte order has to be swapped on decoding
#decoder = BinaryPayloadDecoder.fromRegisters(rr.registers, Endian.Big, wordorder=Endian.Big)
#initialVal = decoder.decode_32bit_float()

#raw = struct.pack('>HH',rr.getRegister(1),getRegister(0))
#value = struct.unpack('>f',raw)[0]


#Importing the dataset 
data = pd.read_csv("/Users/jonnylogue/Desktop/SwanlinbarDataNew.csv", low_memory=False)
#Selecting the relevant data columns from the dataset an d creating a new dataframe to hold the info
df1 = data[['CL001', 'CL002','CLTAR','pH001','TP001','CTTAR','TCTEFF','FL001','TU001','CL_T1_VOL','FL004','LI001','M001','CLDOSE',]]


#Data Cleaning & processing
#changing the data types from objects to numerical floats to allow to be processed through the ML model
df1['CL001'] = pd.to_numeric(df1['CL001'],errors ='coerce')
df1['CL002'] = pd.to_numeric(df1['CL002'],errors ='coerce')
df1['CLTAR'] = pd.to_numeric(df1['CLTAR'],errors ='coerce')
df1['pH001'] = pd.to_numeric(df1['pH001'],errors ='coerce')
df1['TP001'] = pd.to_numeric(df1['TP001'],errors ='coerce')
df1['CTTAR'] = pd.to_numeric(df1['CTTAR'],errors ='coerce')
df1['TCTEFF'] = pd.to_numeric(df1['TCTEFF'],errors ='coerce')
df1['FL001'] = pd.to_numeric(df1['FL001'],errors ='coerce')
df1['TU001'] = pd.to_numeric(df1['TU001'],errors ='coerce')
df1['CLDOSE'] = pd.to_numeric(df1['CLDOSE'],errors ='coerce')
df1['CL_T1_VOL'] = pd.to_numeric(df1['CL_T1_VOL'],errors ='coerce')
df1['FL004'] = pd.to_numeric(df1['FL004'],errors ='coerce')
df1['LI001'] = pd.to_numeric(df1['LI001'],errors ='coerce')

#function to sift through entire dataframe and drop any labels with NaN
#Any labels with NaN will cause errors when trying to run the data sets through the models
df1 = df1.dropna()
#print(df1.describe())

#For testing purposes, after the model has been trained on the dataset then these variables
#can be manually changed and tweeked to run through the model to compare and contrast
#the prediction of the model to what the actual CLDose and CLFlow values were for a given set
#of inputs. 
#Labels order on EXEL inputs C D H I K N O Q S T, outputs L P 
CL001_IP = 1.636
CL002_IP = 1.2712
#CLTAR_IP = 1.65
ph001_IP = 7.4354
TP001_IP = 11.316
#CTTAR_IP = 21
TCTEFF_IP = 18.87565
FL001_IP = 7.488889
TU001_IP = 0.08
CL_T1_VOL_IP = 294
FL004_IP = -7.523839
LI001_IP = 0.420152


#X and y assigned with the data required, X being the independant variable and y being the
#dependant variable
#i.e. our outputs (y) is dependant on everything that is loaded into X 
X = df1[['CL001','CL002','pH001','TP001','TCTEFF','FL001','TU001','CL_T1_VOL','FL004','LI001']]
y = df1[['CLDOSE','M001']]

#dataset is broken up for training, testing and verification
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
#fitting Simple Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
score = regressor.score(X_test,y_test)

#testing the trainging and testing set
Y_pred = regressor.predict(X_train)
y_pred = regressor.predict(X_test)

#Linerar regressor model predicts on the inputable values created above
predict = regressor.predict([[CL001_IP,CL002_IP,ph001_IP,TP001_IP,TCTEFF_IP,FL001_IP,TU001_IP,CL_T1_VOL_IP,FL004_IP,LI001_IP]])
#CLDose and CLFlow pulled apart from the prediction output and seperated for 
#further use elesewhere in the code i.e. to be written back to the PLC.
ClDose_Pred = predict.item(0)
ClFlow_Pred = predict.item(1)
#print(regressor.coef_)

print('------------Linear Regression------------')
print('Predicted CLDose value is %f' %ClDose_Pred)
print('-----------------------------------------')
print('Predicted CLFlow value is %f' %ClFlow_Pred)
print('-----------------------------------------')
print('Model score value is : %.5f' %score)
print('------------Linear Regression------------')




















