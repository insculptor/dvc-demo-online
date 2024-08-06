import os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from dotenv import load_dotenv

load_dotenv()


base_path = Path(os.getenv("BASE_DATA_DIR"))


#### DVCLIVE Tracking ####
from dvclive import Live



df = pd.read_csv(os.path.join(base_path,'student_performance.csv'))

df = df[['gender', 'math score', 'reading score', 'writing score']]
df['gender'] = df['gender'].map({'female': 1, 'male': 0})


X = df[['math score', 'reading score', 'writing score']]
y = df['gender']


print(X.shape,y.shape)
n_estimators = 100
max_depth = 10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

with Live(save_dvc_exp=True) as live:
    live.log_metric("Accuracy",accuracy_score(y_test, y_pred))
    live.log_metric("Precision",precision_score(y_test, y_pred))
    live.log_metric("Recall",recall_score(y_test, y_pred))
    live.log_metric("F1",f1_score(y_test, y_pred))
    
    live.log_param("n_estimators",n_estimators)
    live.log_param("max_depth",max_depth)