# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 18:40:25 2023
@author: Priyanka J 
"""
# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
import seaborn as sns 

# Loading Dataset
milk = pd.read_csv('milknew.csv')
print(milk)
milk.info()
X = milk.drop('Grade', axis=1)
y = milk['Grade']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes model 
nb_classifier = GaussianNB()

# Train the model on the training data
nb_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = nb_classifier.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy= {accuracy:.2f}')

# Printing classification report
print(classification_report(y_test, y_pred))

#Creating and Displaying the Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=nb_classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb_classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#Boxplot
def graph(y):
    sns.boxplot(x="Grade",y=y,data=milk)
    plt.figure(figsize=(10,10)) 
graph('pH') 
graph('Temprature')
graph('Taste')
graph('Odor') 
graph('Turbidity')
graph('Colour')   
plt.show() 

#Creating and Displaying heatmap to show corelations
sns.heatmap(milk.corr(method='pearson'),annot=True) 

#Violin plots for InputFeatures Vs Grade
fig, ax = plt.subplots(figsize =(9, 7))
sns.violinplot(ax = ax, x = milk["Grade"],y = milk["pH"]) 
fig, ax = plt.subplots(figsize =(9, 7))
sns.violinplot(ax = ax, x = milk["Grade"],y = milk["Temprature"])
fig, ax = plt.subplots(figsize =(9, 7))
sns.violinplot(ax = ax, x = milk["Grade"],y = milk["Taste"])
fig, ax = plt.subplots(figsize =(9, 7))
sns.violinplot(ax = ax, x = milk["Grade"],y = milk["Odor"])
fig, ax = plt.subplots(figsize =(9, 7))
sns.violinplot(ax = ax, x = milk["Grade"],y = milk["Turbidity"])
fig, ax = plt.subplots(figsize =(9, 7))
sns.violinplot(ax = ax, x = milk["Grade"],y = milk["Colour"]) 

#Pairplot 
sns.pairplot(data=milk,hue='Grade',height=2)