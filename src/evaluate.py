#Import necessary libraries
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import load_and_clean_data, get_features_and_target, split_data

data_path = '../data/raw/ai4i2020.csv'
model_path = '../models/xgboost_pipeline.joblib'

def evaluate_model():
    df = load_and_clean_data(data_path)
    X, y = get_features_and_target(df)
    
    _, X_test, _, y_test = split_data(X, y)

    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(X_test)

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=['Functional (0)', 'Failure (1)']))

   # Generate and display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Predicted OK', 'Predicted Defect'],
                yticklabels=['Actual OK', 'Actual Defect'])
    plt.title('Confusion Matrix on X_test')
    plt.show()

if __name__ == "__main__":
    evaluate_model()