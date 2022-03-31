# CSE6250 Big Data for Health - Group Project

This repo contains our source code for reproducing the findings in  [Kamal et al., 2020](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-01331-7).

Authors: Rebecca Kee, Na Wu

Your repo should include detailed documents (README file) telling readers:
* Dependencies (which packages are required)
* Download instruction of data and pretrained model (if applicable)
* Functionality of scripts: preprocessing, training, evaluation, etc.
* Instruction to run the code

## Instructions to run code

### 1. Prepare environment and files

First, build the required environment
```
pip install -r requirement.txt
```

Create a folder for MIMIC-III data
``` 
mkdir PAVE/data/mimic-iii-data
```

Put the following MIMIC-III files in `PAVE/data/mimic-iii-data` 
* ADMISSIONS.csv
* CHARTEVENTS.csv
* D_ITEMS.csv
* PATIENTS.csv

### 2. Data preprocessing

Run `process_mimic.py` to transform the raw MIMIC-III data. This will output the following files in PAVE/data:
* demo.csv: patients' demographics
* label.csv: ground truth
* data.csv: temporal records

Create folders for additional data preprocessing output
```
mkdir PAVE/result
mkdir PAVE/data
mkdir PAVE/data/models
```

Generate json files for PAVE model
```
cd PAVE/preprocessing/
python gen_master_feature.py 
python gen_feature_time.py
python gen_feature_order.py 
python gen_vital_feature.py 
python gen_label.py 
```

### 3. Model training

Train and validate the PAVE model, the best model will be saved in PAVE/data/models/
```
cd ../code/
python main.py 
```
### 4. Test model
```
python main.py --phase test --resume PAVE/data/models/best.ckpt
```
