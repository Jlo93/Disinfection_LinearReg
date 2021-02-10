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
import pymodbus
import threading
import time
import random 
import warnings
warnings.filterwarnings("ignore")


from firebase import firebase
from mpl_toolkits.mplot3d import Axes3D
from sklearn import neighbors,datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.client.sync import ModbusTcpClient

firebase = firebase.FirebaseApplication('https://data-793f6-default-rtdb.firebaseio.com/',None)

#Defines the client PLC's IP address for communications
IP = '192.168.120.11'
#Function Establishes connection and checks comms status, returns true if comms = OK
#Reads the %MW5524 location from the PLC's memory this function will be called for each tag.
def PLC_Tasks(IP_Address,memory_Word,data_Length):

	client = ModbusTcpClient(IP_Address)
	connection_status = client.connect()
	print('---------------Comms Status--------------')
	print('Communication Status with PLC:',connection_status)
	print('---------------Comms Status--------------\n')

	rr = client.read_holding_registers(memory_Word,data_Length) #>rr = client.read_holdiing_registers(5524,2)< worked last test
	#Sets up the decoder to decode the chosen memory word, endian is set to big as the M340 byte order has to be swapped on decoding
	decoder = BinaryPayloadDecoder.fromRegisters(rr.registers, Endian.Big, wordorder=Endian.Big)
	initialVal = decoder.decode_32bit_float()
	raw = struct.pack('>HH',rr.getRegister(1),getRegister(0))
	value = struct.unpack('>f',raw)[0]

	print(value)



#PLC_Tasks(IP,5524,2)


#Function to generate instrumentation data for testing live algorithim & database handling.
def data_generator():

	CL001 = random.uniform(1.4,1.65)
	CL002 = random.uniform(1.3,1.5)
	ph001 = random.uniform(10.5,8.4)
	TP001 = random.uniform(11.5,11.9)
	TCTEFF = random.uniform(120.0,125.0)
	FL001 = random.uniform(7.8,8.3)
	TU001 = random.uniform(0.075,0.08)
	CL_T1_VOL = random.uniform(318,320)
	FL004 = random.uniform(7.8,8.2)
	LI001 = random.uniform(0.45,0.47)

	return (CL001,CL002,ph001,TP001,TCTEFF,FL001,TU001,CL_T1_VOL,FL004,LI001)

#Function to continuously update the real-time data base values every 5 seconds.
def Database_Update(tcteff,ph001,tp001,cl001,cl002,ClDose_Pred,ClFlow_Pred,tu001,fl001,fl004,li001):
	threading.Timer(5.0, Database_Update).start()

	result = firebase.put('/CML_Data/Swanlinbar_Data_Points/','Contact Time(TCTEFF)',tcteff) #update data
	result = firebase.put('/CML_Data/Swanlinbar_Data_Points/','pH',ph001)
	result = firebase.put('/CML_Data/Swanlinbar_Data_Points/','Temp',tp001) #update data
	result = firebase.put('/CML_Data/Swanlinbar_Data_Points/','Cl Res CL001',cl001) #update data
	result = firebase.put('/CML_Data/Swanlinbar_Data_Points/','Cl Val CL002',cl002) #update data
	result = firebase.put('/CML_Data/Swanlinbar_Data_Points/','Cldose rate Prediction',ClDose_Pred) #update data
	result = firebase.put('/CML_Data/Swanlinbar_Data_Points/','Clflow rate Prediction',ClFlow_Pred) #update data
	result = firebase.put('/CML_Data/Swanlinbar_Data_Points/','Turbidity TU001',tu001) #update data
	result = firebase.put('/CML_Data/Swanlinbar_Data_Points/','Inflow FL001',fl001) #update data
	result = firebase.put('/CML_Data/Swanlinbar_Data_Points/','Outlet Flow FL004',fl004) #update data
	result = firebase.put('/CML_Data/Swanlinbar_Data_Points/','Reservoir Level LI001',li001) #update data


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

aveData = df1.mean(axis = 0)


#For testing purposes, after the model has been trained on the dataset then these variables
#can be manually changed and tweeked to run through the model to compare and contrast
#the prediction of the model to what the actual CLDose and CLFlow values were for a given set
#of inputs. 
#Labels order on EXEL inputs C D H I K N O Q S T, outputs L P 



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

while True:

	CL001,CL002,ph001,TP001,TCTEFF,FL001,TU001,CL_T1_VOL,FL004,LI001 = data_generator()

	CL001_IP = CL001
	CL002_IP = CL002
	#CLTAR_IP = 1.65
	ph001_IP = ph001
	TP001_IP = TP001
	#CTTAR_IP = 21
	TCTEFF_IP = TCTEFF
	FL001_IP = FL001
	TU001_IP = TU001
	CL_T1_VOL_IP = CL_T1_VOL
	FL004_IP = FL004
	LI001_IP = LI001

	#Linerar regressor model predicts on the inputable values created above
	predict = regressor.predict([[CL001_IP,CL002_IP,ph001_IP,TP001_IP,TCTEFF_IP,FL001_IP,TU001_IP,CL_T1_VOL_IP,FL004_IP,LI001_IP]])
	#CLDose and CLFlow pulled apart from the prediction output and seperated for 
	#further use elesewhere in the code.
	ClDose_Pred = predict.item(0)
	ClFlow_Pred = predict.item(1)
	#print(regressor.coef_)

	#print('------------Linear Regression------------')
	#print('Predicted CLDose value is %f' %ClDose_Pred)
	#print('-----------------------------------------')
	#print('Predicted CLFlow value is %f' %ClFlow_Pred)
	#print('-----------------------------------------')


	Database_Update(tcteff=TCTEFF_IP,ph001=ph001_IP,tp001=TP001_IP,
		cl001=CL001_IP,cl002=CL002_IP,ClDose_Pred=ClDose_Pred,ClFlow_Pred=ClFlow_Pred,
		tu001=TU001_IP,fl001=FL001_IP,fl004=FL004_IP,li001=LI001_IP)














