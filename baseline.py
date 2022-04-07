import os
from PAVE.code.tools import parse, py_op
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
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
    X_df = demo.merge(minmax_df, on='id')
    
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
    y_train = label[np.in1d(label['id'].values, patient_train)]
    y_valid = label[np.in1d(label['id'].values, patient_valid)]
    y_test = label[np.in1d(label['id'].values, patient_test)]
    return X_train, X_valid, X_test, y_train, y_valid, y_test


# def logistic_regression(X_train, Y_train, X_test):
#     logit_clf = LogisticRegression(random_state=args.seed).fit(X_train, Y_train)
#     Y_pred = logit_clf.predict(X_test)
#     return Y_pred

# def svm(X_train, Y_train, X_test):
#     svm_clf = LinearSVC(random_state=args.seed).fit(X_train, Y_train)
#     Y_pred = svm_clf.predict(X_test)
#     return Y_pred

# def random_forest(X_train, Y_train, X_test):
#     tree_clf = DecisionTreeClassifier(random_state=args.seed, max_depth=5).fit(X_train, Y_train)
#     Y_pred = tree_clf.predict(X_test)
#     return Y_pred

# def classification_metrics(Y_pred, Y_true):
#     accuracy = float(accuracy_score(Y_true, Y_pred))
#     auc = float(roc_auc_score(Y_true, Y_pred))
#     precision = float(precision_score(Y_true, Y_pred))
#     recall = float(recall_score(Y_true, Y_pred))
#     f1score = float(f1_score(Y_true, Y_pred))
#     return accuracy,auc,precision,recall,f1score

# def display_metrics(classifierName,Y_pred,Y_true):
# 	print("______________________________________________")
# 	print(("Classifier: "+classifierName))
# 	acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
# 	print(("Accuracy: "+str(acc)))
# 	print(("AUC: "+str(auc_)))
# 	print(("Precision: "+str(precision)))
# 	print(("Recall: "+str(recall)))
# 	print(("F1-score: "+str(f1score)))
# 	print("______________________________________________")
# 	print("")

# def main():
    # prep_data(data, demo, label)
    # display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train,X_test),Y_test)
	# display_metrics("SVM",svm_pred(X_train,Y_train,X_test),Y_test)
	# display_metrics("Random Forest",decisionTree_pred(X_train,Y_train,X_test),Y_test)

# if __name__ == '__main__':
#     main()