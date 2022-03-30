import pandas as pd
import numpy as np

# Load data
filepath = "PAVE/data/mimic-iii-data/"
admissions = pd.read_csv(filepath + 'ADMISSIONS.csv', usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'HOSPITAL_EXPIRE_FLAG'])
d_items = pd.read_csv(filepath + 'D_ITEMS.csv', usecols=['ITEMID', 'LABEL'])
patients = pd.read_csv(filepath + 'PATIENTS.csv', usecols=['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'EXPIRE_FLAG'])
events = pd.read_csv(filepath + 'CHARTEVENTS.csv', usecols=['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUENUM'], dtype={'SUBJECT_ID':int, 'HADM_ID':int, 'ITEMID':int, 'CHARTTIME':str, 'VALUE':str, 'VALUENUM':float})

# Inclusion/exclusion lists of strings for selected variables
var_list = ['diastolic', 'glucose', 'heart rate', 'blood pressure mean', 'bp mean', 'respiratory rate', 'spo2', 'o2 saturation', 'systolic', 'temperature c']
exclude_list = ['signal', 'alarm', 'desat limit', 'unloading', 'control', 'set', 'spontaneous', 'score', 'soft', 'apacheiv', 'rvsystolic', 'monitor #', 'blood temperature', 'orthostat', 'lowest', 'urine', 'iabp', 'femoral']

# Filter selected variables in d_items
d_varitems = d_items.copy()
d_varitems['LABEL'] = d_varitems['LABEL'].astype(str).str.lower()
d_varitems = d_varitems[(d_varitems['LABEL'].str.contains('|'.join(var_list), regex=True)) & (~d_varitems['LABEL'].str.contains('|'.join(exclude_list), regex=True))]

# Add column for label groupings
def recode_label(row):
    if "heart rate" in row['LABEL']:
        return 'heartrate'
    elif "diastolic" in row['LABEL']:
        return 'diasbp'
    elif "systolic" in row['LABEL']:
        return 'sysbp'
    elif "mean" in row['LABEL']:
        return 'meanbp'
    elif "respiratory rate" in row['LABEL']:
        return 'resprate'
    elif "temperature" in row['LABEL']:
        return 'tempc'
    elif "o2" in row['LABEL']:
        return 'spo2'
    else:
        return 'glucose'
    
d_varitems['LABEL_GRP'] = d_varitems.apply(recode_label, axis=1)

# Filter selected variables in chart events, drop NAs
events_f = events.copy()
events_f = events_f[(events_f['ITEMID'].isin(d_varitems['ITEMID'])) & (~events_f['VALUENUM'].isna())]

# Keep only most recent admission per patient
admissions_f = admissions.copy()
admissions_f['ADMITTIME'] = admissions_f['ADMITTIME'].apply(pd.to_datetime, errors='coerce')
admissions_f = admissions_f.sort_values(['SUBJECT_ID', 'ADMITTIME'], ascending=False).groupby('SUBJECT_ID').first().reset_index()

# Merge dfs
merged_df = admissions_f.merge(events_f, on=['SUBJECT_ID','HADM_ID']).merge(d_varitems, on='ITEMID').merge(patients, on='SUBJECT_ID')
merged_df.columns = map(str.lower, merged_df.columns)

# Convert dtypes
merged_df[['charttime', 'admittime', 'dob', 'dod']] = merged_df[['charttime', 'admittime', 'dob', 'dod']].apply(pd.to_datetime, errors='coerce')

# Compute age
merged_df['age'] = ((merged_df['admittime'].values-merged_df['dob'].values).astype(int)/8.64e13//365).astype(int)

# Drop rows with age=0
merged_df = merged_df[merged_df['age']>0]

# Create PAVE input csv files
demo_df = merged_df[['subject_id', 'gender', 'age']].rename(columns={"subject_id": "id"}).drop_duplicates().reset_index(drop=True)
label_df = merged_df[['subject_id', 'hospital_expire_flag']].rename(columns={"subject_id": "id", "hospital_expire_flag": "label"}).drop_duplicates().reset_index(drop=True)
data_df = merged_df[['subject_id', 'charttime', 'label_grp', 'valuenum']].rename(columns={"subject_id": "id", "charttime": "time"}).drop_duplicates()
data_df_wide = pd.pivot_table(data_df, index=['id', 'time'], values='valuenum', columns=['label_grp'], aggfunc=np.mean).reset_index()
  
demo_df.to_csv('PAVE/data/demo.csv', index=False)  
label_df.to_csv('PAVE/data/label.csv', index=False)  
data_df_wide.to_csv('PAVE/data/data.csv', index=False)  

