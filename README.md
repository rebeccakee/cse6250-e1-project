# CSE6250 Big Data for Health - Group Project

This repo contains our source code for reproducing the findings in  [Kamal et al., 2020](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-01331-7).

Authors: Rebecca Kee, Na Wu

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
