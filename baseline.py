import os
from PAVE.code.tools import parse, py_op
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *


args = parse.args
filepath = "PAVE/data/"
print("Loading data...")
data = pd.read_csv(filepath + 'data.csv')
demo = pd.read_csv(filepath + 'demo.csv')
label = pd.read_csv(filepath + 'label.csv')
patient_time_dict = py_op.myreadjson(os.path.join('PAVE/code/', args.result_dir, 'patient_time_dict.json'))

def prep_data(data, demo, label):
    print("Preparing data...")
    # Filter events given a hold-off prediction window
    # Include only events within last 48 hours, excluding those in hold-off prediction window
    data_inwindow = data.copy()
    data_inwindow['time'] = pd.to_datetime(data_inwindow['time'], format='%Y-%m-%d %H:%M:%S')
    max_times = data_inwindow.groupby('id').agg(max_time = ('time', 'max')).reset_index() # Get latest timestamp per patient
    data_inwindow = data_inwindow.merge(max_times, on='id') 
    data_inwindow['timerange_start'] = data_inwindow['max_time']-timedelta(hours=48) # Get timestamp of 48hr before max time (start of inclusion window)
    data_inwindow = data_inwindow[data_inwindow['time']>=data_inwindow['timerange_start']] # Include only events within last 48 hours
    data_inwindow['timerange_end'] = data_inwindow['max_time']-timedelta(hours=-args.last_time) 
    data_inwindow = data_inwindow[data_inwindow['time']<=data_inwindow['timerange_end']] # Exclude events in prediction window
    data_inwindow = data_inwindow.drop(columns=['max_time', 'timerange_start', 'timerange_end'])
    
    # Extract min & max of each variable, merge with patient demo
    minmax_df = data_inwindow.loc[:,data_inwindow.columns!='time'].groupby('id').agg(['min', 'max'])
    minmax_df.columns = ["_".join(col).rstrip('_') for col in minmax_df.columns.to_flat_index()]
    X_df = demo.merge(minmax_df, on='id').dropna() # Drop rows with any NAs
    X_df['gender'] = X_df['gender'].replace({'M': 1, 'F': 0}) # All columns need to be numeric for baseline methods

    # Split patients into train/valid/test according to splits in PAVE model
    patient_time_record_dict = py_op.myreadjson('PAVE/result/patient_time_record_dict.json')
    patient_master_dict = py_op.myreadjson('PAVE/result/patient_master_dict.json')
    patient_label_dict = py_op.myreadjson('PAVE/result/patient_label_dict.json')
    patients = sorted(set(patient_label_dict.keys()) & set(patient_time_record_dict) & set(patient_master_dict))
    n_train = int(0.7 * len(patients))
    n_valid = int(0.2 * len(patients))
    patient_train = patients[:n_train]
    patient_valid = patients[n_train:n_train+n_valid]
    patient_test  = patients[n_train+n_valid:]
    
    X_train = X_df[np.in1d(X_df['id'].values, patient_train)]
    X_valid = X_df[np.in1d(X_df['id'].values, patient_valid)]
    X_test = X_df[np.in1d(X_df['id'].values, patient_test)]
    y_train = label[np.in1d(label['id'].values, X_train['id'])].label
    y_valid = label[np.in1d(label['id'].values, X_valid['id'])].label
    y_test = label[np.in1d(label['id'].values, X_test['id'])].label
    print("Data preparation done!")
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def logistic_regression(X_train, y_train, X_test):
    grid = {"solver":['lbfgs','newton-cg','liblinear','sag','saga'], "C":np.arange(0.5, 1.0, 0.1)}
    model = LogisticRegression(random_state=args.seed, max_iter=4000)
    print("Logistic regression - Performing 10-fold CV...")
    cv = GridSearchCV(model, grid, scoring='roc_auc', cv=10).fit(X_train, y_train)
    print("Logistic regression - Best parameters:", cv.best_params_)
    print("Logistic regression - Best cv auc:", cv.best_score_)
    print("Logistic regression - Fitting best model...")
    best_model = LogisticRegression(random_state=args.seed, solver=cv.best_params_['solver']).fit(X_train, y_train)
    print("Logistic regression - Predicting on test set...")
    Y_pred = best_model.predict(X_test)
    print("Logistic regression - Performance on test set:")
    return Y_pred

def svm(X_train, y_train, X_test):
    grid = {"kernel":['poly','linear','rbf','sigmoid'], "C":np.arange(0.5, 1.0, 0.1)}
    model = SVC(random_state=args.seed)
    print("SVM - Performing 10-fold CV...")
    cv = GridSearchCV(model, grid, scoring='roc_auc', cv=10).fit(X_train, y_train)
    print("SVM - Best parameters:", cv.best_params_)
    print("SVM - Best cv auc:", cv.best_score_)
    print("SVM - Fitting best model...")
    best_model = SVC(random_state=args.seed, kernel=cv.best_params_['kernel']).fit(X_train, y_train)
    print("SVM - Predicting on test set...")
    Y_pred = best_model.predict(X_test)
    print("SVM - Performance on test set:")
    return Y_pred

def random_forest(X_train, y_train, X_test):
    grid = {"max_depth":[3, 5, 7, 9], "max_features":['sqrt','log2']}
    model = RandomForestClassifier(random_state=args.seed, n_estimators=1000)
    print("RF - Performing 10-fold CV...")
    cv = GridSearchCV(model, grid, scoring='roc_auc', cv=10).fit(X_train, y_train)
    print("RF - Best parameters:", cv.best_params_)
    print("RF - Best cv auc:", cv.best_score_)
    print("RF - Fitting best model...")
    best_model = RandomForestClassifier(random_state=args.seed, n_estimators=1000, max_depth=cv.best_params_['max_depth'], max_features=cv.best_params_['max_features']).fit(X_train, y_train)
    print("RF - Predicting on test set...")
    Y_pred = best_model.predict(X_test)
    print("RF - Performance on test set:")
    return Y_pred

def classification_metrics(Y_pred, Y_true):
    accuracy = float(accuracy_score(Y_true, Y_pred))
    auc = float(roc_auc_score(Y_true, Y_pred))
    precision = float(precision_score(Y_true, Y_pred))
    recall = float(recall_score(Y_true, Y_pred))
    f1score = float(f1_score(Y_true, Y_pred))
    return accuracy,auc,precision,recall,f1score

def display_metrics(classifierName,Y_pred,Y_true):
	print("______________________________________________")
	print(("Classifier: "+classifierName))
	acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
	print(("Accuracy: "+str(acc)))
	print(("AUC: "+str(auc_)))
	print(("Precision: "+str(precision)))
	print(("Recall: "+str(recall)))
	print(("F1-score: "+str(f1score)))
	print("______________________________________________")
	print("")

def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = prep_data(data, demo, label)
    display_metrics("Logistic Regression", logistic_regression(X_train, y_train, X_test), y_test)
    display_metrics("Random Forest", random_forest(X_train,y_train,X_test),y_test)
    display_metrics("SVM", svm(X_train,y_train,X_test),y_test)

if __name__ == '__main__':
    main()
