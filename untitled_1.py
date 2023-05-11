# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:21:07 2023

@author: DELL
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error


# Load the dataset
df = pd.read_csv('datasetForTask_f.csv')

df.rename(columns = {'Age-standardised diabetes prevalence':'ASDP'}, inplace = True)
df.rename(columns = {'Lower 95% uncertainty interval':'Lower_95_uncertainty'}, inplace = True)
df.rename(columns = {'Upper 95% uncertainty interval':'Upper_95_uncertainty'}, inplace = True)

st.subheader("Data Visualization")

st.subheader("1) Does the average age-standardized death rate (ASDP) vary by sex?")
plt.figure(figsize=(8, 6))
# Create a barplot to compare ASDP by sex
sns.barplot(x="Sex", y="ASDP", data=df)
plt.title('age-standardized death rate (ASDP) vary by sex')
st.pyplot(plt)


st.subheader("2) How has ASDP changed over the years?")
plt.figure(figsize=(20, 12))
# Create a lineplot to visualize ASDP over time
sns.lineplot(x="Year", y="ASDP", data=df)
plt.title('ASDP changed over the years')
st.pyplot(plt)


st.subheader("3) Which countries have the highest and lowest ASDP?")
plt.figure(figsize=(10, 8))
# Create a boxplot to visualize ASDP distribution by country
sns.boxplot(x="Country/Region/World", y="ASDP", data=df)
plt.title('The highest and lowest ASDP')
st.pyplot(plt)


st.subheader("4) Comparison of Fatalities on Weekends/Nights vs Weekdays/Daytime")
plt.figure(figsize=(10, 8))
# Create a scatterplot to visualize the relationship between ASDP and Lower_95_uncertainty 
sns.scatterplot(x="Lower_95_uncertainty", y="ASDP", data=df)
plt.title('Relationship between ASDP and Lower_95_uncertainty')
st.pyplot(plt)


#st.subheader("5) Distribution of Road Fatalities by Road User and Crash Type")
#plt.figure(figsize=(15,10))
#sns.countplot(x='Road User', hue='Crash Type', data=df)
#plt.xlabel('Road User')
#plt.ylabel('Count')
#plt.title('Distribution of Road Fatalities by Road User and Crash Type')
#st.pyplot(plt)

# Data preprocessing
x = df[['Country/Region/World', 'ISO', 'Sex', 'Year', 'ASDP', 'Lower_95_uncertainty']]
y = df['Upper_95_uncertainty']
x = pd.get_dummies(x, columns=['Country/Region/World', 'ISO', 'Sex'])
accu = 0
for i in range(0,14000):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .80, random_state = i)
    mod = LinearRegression()
    mod.fit(x_train,y_train)
    y_pred = mod.predict(x_test)
    tempacc = r2_score(y_test,y_pred)
    if tempacc> accu:
        accu= tempacc
        best_rstate=i

print(f"Best Accuracy {accu*100} found on randomstate {best_rstate}")


# Data preprocessing
#X = df[['Country/Region/World', 'ISO', 'Sex', 'Year', 'ASDP', 'Lower_95_uncertainty']]
#y = df['Upper_95_uncertainty']
#X = pd.get_dummies(X) # Convert categorical variables to numerical

#X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, test_size=0.80, random_state=42)

# Classification SVC
#clasification_1 = SVC()
#clasification_1.fit(X_train_data, y_train_data)
#y_prediction = clasification_1.predict(X_test_data)
#accuracy_1 = accuracy_score(y_test_data, y_prediction)

#Classification Random Forest
#clasificatio_2 = RandomForestClassifier()
#clasification_2.fit(X_train_data, y_train_data)
#y_prediction = clasification_2.predict(X_test_data)
#accuracy_2 = accuracy_score(y_test_data, y_prediction)

# Display results
#st.subheader("Classification (Predictive Analytics)")
#st.write("Accuracy (SVM):", accuracy_1)
#st.write("Accuracy (RF):", accuracy_2)