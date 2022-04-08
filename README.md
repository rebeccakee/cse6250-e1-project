# CSE6250 Big Data for Health - Group Project

Authors: Rebecca Kee, Na Wu

This repo contains our source code for reproducing the Pattern Attention with Value Embedding (PAVE) model and the findings described in [Kamal et al., 2020](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-01331-7).

## Repo contents
* The `PAVE/` subfolder contains all the necessary scripts for training and evaluating the PAVE model. 
* The `process_mimic.py` script prepares the raw MIMIC-III data so that it is suitable for loading into the PAVE model. 
* The `baseline.py` script implements the training of 3 baseline models for comparing the performance of the PAVE model to. The baseline methods include Logistic Regression, Random Forest, and Support Vector Machine (SVM). It also further processes the MIMIC-III data so that is it feasible for fitting the baseline models.

## Required data 
MIMIC-III data is required to run our source code. You may apply for access to MIMIC-III data following the instructions [here](https://mimic.mit.edu/docs/gettingstarted/). 

## Instructions to run code

### 1. Prepare environment and files

1. Build the required environment
```
pip install -r requirement.txt
```

2. Create various folders to store data and output
``` 
mkdir PAVE/data/mimic-iii-data
mkdir PAVE/result
mkdir PAVE/data
mkdir PAVE/data/models
```

3. Put the following MIMIC-III files in `PAVE/data/mimic-iii-data` 
* ADMISSIONS.csv
* CHARTEVENTS.csv
* D_ITEMS.csv
* PATIENTS.csv

### 2. Data preprocessing

1. Run `process_mimic.py` to transform the raw MIMIC-III data 
```
python process_mimic.py
```
This will output the following files in PAVE/data:
* demo.csv: patients' demographics
* label.csv: ground truth
* data.csv: temporal records

2. Save input data as json files for PAVE model
```
cd PAVE/preprocessing/
python gen_master_feature.py 
python gen_feature_time.py
python gen_feature_order.py 
python gen_vital_feature.py 
python gen_label.py 
```

### 3. PAVE model training

Train and validate the PAVE model, the best model will be saved in PAVE/data/models/
```
cd ../code/
python main.py 
```
### 4. Evaluate PAVE model on test set
```
python main.py --phase test --resume PAVE/data/models/best.ckpt
```
### 5. Train baseline models and evaluate on test set
```
cd ../..
python baseline.py
```
