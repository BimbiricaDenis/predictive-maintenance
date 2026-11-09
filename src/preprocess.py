#Libraries
import pandas as pd
from sklearn.model_selection import train_test_split

#load and clean data
def load_and_clean_data(filepath):
  data = pd.read_csv(filepath)
  data.drop(columns = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], inplace=True)
  return data

#Separate features and target variable
def get_features_and_target(data):
  X = data.drop(columns = ['Machine failure'])
  y = data['Machine failure']
  return X, y

#Split data into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,stratify=y)
    return X_train, X_test, y_train, y_test