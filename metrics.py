# Python script for confusion matrix creation. 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random
from math import sqrt
import pandas as pd
import numpy as np
df=pd.read_csv('C:\\Users\\chait\\Desktop\\Video_Time\\testing.csv')
#print(df.head())
class_original=df['Class']
print(class_original.head())
class_pred=df['Predicted Class']
print(class_pred.head())
actual = class_original
predicted = class_pred
results = confusion_matrix(actual, predicted) 
print('Confusion Matrix :')
print(results)
print ('Accuracy Score :',accuracy_score(actual, predicted) )
print ('Report : ')
print (classification_report(actual, predicted) )
coefficient_of_determination = r2_score(actual,predicted)
print("R2 error: ", coefficient_of_determination)
meanSquaredError=mean_squared_error(actual, predicted)
print("MSE:", meanSquaredError)
rootMeanSquaredError = sqrt(meanSquaredError)
print("RMSE:", rootMeanSquaredError)
