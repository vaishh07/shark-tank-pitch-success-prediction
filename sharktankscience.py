import pandas as pd
import datetime 
#import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('SHARKTANK.csv')

# Data Cleaning
def clean_data(df) :
     
    # Making 'Company website' & 'Started in' columns relevent to the model
    df['website'] = df['Company Website'].notna().astype(int)
    current_year = datetime.datetime.now().year
    df['years_active'] = current_year - df['Started in']

    # Cleaning the null values
    # filling 0 in place of NaN
    null_columns = ['Yearly Revenue','Monthly Sales','Net Margin','Total Deal Amount','Total Deal Equity','Total Deal Debt','Deal Valuation','Has Patents','Accepted Offer']
    df[null_columns] = df[null_columns].fillna(0)

    # filling mean margin in Nan for Gross Margin
    mean_margin = df['Gross Margin'].mean()
    df['Gross Margin'] = df['Gross Margin'].fillna(mean_margin)

    # filling median value in Nan for years_active
    median_years = df['years_active'].median()
    df['years_active'] = df['years_active'].fillna(median_years)

    # Changing the datatype
    df['Has Patents'] = df['Has Patents'].astype(int)
    df.dropna(inplace=True)
    # Dropping the irrelevent columns
    df = df.drop(['Net Margin','Pitchers Average Age','Company Website','Started in','Pitchers State','Startup Name', 'Business Description','Total Deal Amount','Total Deal Equity','Total Deal Debt','Deal Valuation'],axis=1)
    
    return df

def train_model(df):
    # One-hot encode 'Industry'
    # df = pd.get_dummies(df, columns=['Industry'])
    
    
    # Drop target variables and split data into features (X) and target variable (y)
    X = df.drop(['Received Offer', 'Accepted Offer','Industry'], axis=1)
    y = df['Received Offer']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the logistic regression model
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    
    # Predict on the test set and calculate accuracy
    y_pred = lr.predict(X_test)
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)
    
    return lr