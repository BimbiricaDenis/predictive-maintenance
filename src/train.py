#Import necessary libraries
from preprocess import load_and_clean_data, get_features_and_target, split_data
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import GridSearchCV

#Paths for the data and model
data_path = '../data/raw/ai4i2020.csv'
model_path = '../models/xgboost_pipeline.joblib'

def train_model():
    df = load_and_clean_data(data_path)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    #Define the preprocessing for numerical and categorical features
    numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    categorical_features = ['Type']

    # Define the preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', StandardScaler(), numerical_features),
            ('cat', OrdinalEncoder(categories=[['L', 'M', 'H']]), categorical_features)
        ]
    )

    class_report = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    # Define the pipeline with the preprocessor and the XGBoost model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(scale_pos_weight=class_report,random_state = 42))
    ])

    # Define the hyperparameter grid for tuning
    param_grid = {
        'model__n_estimators': [100, 200,300],
        'model__max_depth': [3, 5,7],
        'model__learning_rate': [0.01, 0.1,0.2]
    }

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='recall',verbose=1)

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    joblib.dump(grid_search.best_estimator_, model_path)

if __name__ == "__main__":
    train_model()