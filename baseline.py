import os
from PAVE.code.tools import parse, py_op
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

args = parse.args
filepath = "PAVE/data/"
data = pd.read_csv(filepath + 'data.csv')
demo = pd.read_csv(filepath + 'demo.csv')
label = pd.read_csv(filepath + 'label.csv')

def prep_data(data, demo, label):
    # Extract min & max of each variable, merge with patient demo
    minmax_df = data.loc[:,data.columns!='time'].groupby('id').agg(['min', 'max'])
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
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def logistic_regression(X_train, y_train, X_test):
    grid = {"solver":['lbfgs','newton-cg','liblinear','sag','saga']}
    model = LogisticRegression(random_state=args.seed, max_iter=1000)
    print("Logistic regression - Performing 10-fold CV...")
    cv = GridSearchCV(model, grid, scoring='roc_auc', cv=10).fit(X_train, y_train)
    print("Logistic regression - Best parameters:", cv.best_params_)
    print("Logistic regression - Best cv auc:", cv.best_score_)
    print("Logistic regression - Fitting best model...")
    best_model = LogisticRegression(random_state=args.seed, solver=cv.best_params_['solver']).fit(X_train, y_train)
    Y_pred = best_model.predict(X_test)
    print("Logistic regression fitting - Done!")
    return Y_pred

def svm(X_train, y_train, X_test):
    grid = {"kernel":['poly','linear','rbf','sigmoid']}
    model = SVC(random_state=args.seed)
    print("SVM - Performing 10-fold CV...")
    cv = GridSearchCV(model, grid, scoring='roc_auc', cv=10).fit(X_train, y_train)
    print("SVM - Best parameters:", cv.best_params_)
    print("SVM - Best cv auc:", cv.best_score_)
    print("SVM - Fitting best model...")
    best_model = SVC(random_state=args.seed, kernel=cv.best_params_['kernel']).fit(X_train, y_train)
    Y_pred = best_model.predict(X_test)
    print("SVM fitting - Done!")
    return Y_pred

# def random_forest(X_train, Y_train, X_test):
#     tree_clf = DecisionTreeClassifier(random_state=args.seed, max_depth=5).fit(X_train, Y_train)
#     Y_pred = tree_clf.predict(X_test)
#     return Y_pred

# X_train, X_valid, X_test, y_train, y_valid, y_test = prep_data(data, demo, label)
# Y_pred = random_forest(X_train, y_train, X_test)

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

# def main():
    # prep_data(data, demo, label)
    # display_metrics("Logistic Regression", logistic_regression(X_train, y_train, X_test), y_test)
	# display_metrics("SVM", svm(X_train,y_train,X_test),y_test)
	# display_metrics("Random Forest", random_forest(X_train,y_train,X_test),y_test)

# if __name__ == '__main__':
#     main()