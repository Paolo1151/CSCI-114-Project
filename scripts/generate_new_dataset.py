import os
import sys
sys.path.append('..')

import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg

def partition_dataset(df, num_a=150, num_b=150, num_c=150, val_a=1.0, val_b=0.0, val_c=2.0):
    df_a = df[df.iloc[:,0] == val_a]
    df_b = df[df.iloc[:,0] == val_b]
    df_c = df[df.iloc[:,0] == val_c]

    mvn = min(df_a.shape[0], df_b.shape[0], df_c.shape[0])

    df_a = df_a.sample(mvn)
    df_b = df_b.sample(mvn)
    df_c = df_c.sample(mvn)

    frames = [df_a, df_b, df_c]
    df = pd.concat(frames)
    
    return df

file_raw_data = pd.read_csv(os.path.join(cfg.DATA_PATH, 'cleaned.csv'))

# data pre-processing steps
# step 1: remove null and unknown values from each column
file_raw_data = file_raw_data.dropna()
for col in file_raw_data.columns:
  file_raw_data = file_raw_data.drop(file_raw_data.loc[file_raw_data[col] == 'Unknown'].index)
  file_raw_data = file_raw_data.drop(file_raw_data.loc[file_raw_data[col] == 'unknown'].index)
# step 2: merge jhs with hs in educ level, remove '1'
file_raw_data['Educational_level'] = file_raw_data['Educational_level'].replace(['Junior high school'], 'High school')
# step 3: merge Darkness - no lighting and Darkness - lights unlit in Light_conditions
file_raw_data['Light_conditions'] = file_raw_data['Light_conditions'].replace(['Darkness - no lighting'], 'Darkness')
file_raw_data['Light_conditions'] = file_raw_data['Light_conditions'].replace(['Darkness - lights unlit'], 'Darkness')
# step 4: merge 'Raining' and 'windy' and 'raining and windy' to raining and/or windy
file_raw_data['Weather_conditions'] = file_raw_data['Weather_conditions'].replace(['Raining'], 'Raining and/or Windy')
file_raw_data['Weather_conditions'] = file_raw_data['Weather_conditions'].replace(['Raining and Windy'], 'Raining and/or Windy')
file_raw_data['Weather_conditions'] = file_raw_data['Weather_conditions'].replace(['Windy'], 'Raining and/or Windy')
# step 5: Vehicle_movement: merge moving back and reverse
file_raw_data['Vehicle_movement'] = file_raw_data['Vehicle_movement'].replace(['Moving Backward'], 'Reversing')
# step 6: cause of accident: changing lane to left and right combine to just changing lane
file_raw_data['Cause_of_accident'] = file_raw_data['Cause_of_accident'].replace(['Changing lane to the left'], 'Changing lane')
file_raw_data['Cause_of_accident'] = file_raw_data['Cause_of_accident'].replace(['Changing lane to the right'], 'Changing lane')
# step 7: cause of accident: merge drunk driving and drug driving to driving under the influence, merge driving fast and overspeed
file_raw_data['Cause_of_accident'] = file_raw_data['Cause_of_accident'].replace(['Driving under the influence of drugs'], 'Driving under the influence')
file_raw_data['Cause_of_accident'] = file_raw_data['Cause_of_accident'].replace(['Drunk driving'], 'Driving under the influence')
file_raw_data['Cause_of_accident'] = file_raw_data['Cause_of_accident'].replace(['Overspeed'], 'Driving at high speed')
# step 8: pedestrian movement: merge 'crossing from...' into one, merge 'walking along in carriage...' into one, merge 'In carriageway,...' into 'In carriageway, not crossing'
file_raw_data['Pedestrian_movement'] = file_raw_data['Pedestrian_movement'].replace(['Crossing from driver\'s nearside'], 'Crossing')
file_raw_data['Pedestrian_movement'] = file_raw_data['Pedestrian_movement'].replace(['Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle'], 'Crossing')
file_raw_data['Pedestrian_movement'] = file_raw_data['Pedestrian_movement'].replace(['Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle'], 'Crossing')
file_raw_data['Pedestrian_movement'] = file_raw_data['Pedestrian_movement'].replace(['In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)'], 'In carriageway, not crossing')
file_raw_data['Pedestrian_movement'] = file_raw_data['Pedestrian_movement'].replace(['In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle'], 'In carriageway, not crossing')
file_raw_data['Pedestrian_movement'] = file_raw_data['Pedestrian_movement'].replace(['Walking along in carriageway, back to traffic'], 'Walking along in carriageway with traffic')
file_raw_data['Pedestrian_movement'] = file_raw_data['Pedestrian_movement'].replace(['Walking along in carriageway, facing traffic'], 'Walking along in carriageway with traffic')

df = pd.get_dummies(file_raw_data ,columns = file_raw_data.columns.difference(['Accident_severity']))

df = df.reset_index().drop(['index'], axis=1)


training = partition_dataset(df)

X = training.drop(columns=['Accident_severity'])
y = training['Accident_severity']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

df_train = X_train.copy()
df_train['Accident_severity'] = y_train

df_test = X_test.copy()
df_test['Accident_severity'] = y_test

# Saving Datasets
df_train.to_csv(os.path.join(cfg.DATA_PATH, 'cleaned_train.csv'), index=False)
df_test.to_csv(os.path.join(cfg.DATA_PATH, 'cleaned_test.csv'), index=False)