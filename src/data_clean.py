import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

df = pd.read_csv('data/churn_train.csv',parse_dates=['last_trip_date','signup_date'])
date_cutoff = datetime.date(2014, 6, 1)

(df.last_trip_date < date_cutoff).value_counts()


df.info()
def load_data():
    df = pd.read_csv('data/churn_train.csv',parse_dates=['last_trip_date','signup_date'])
    date_cutoff = datetime.date(2014, 6, 1)
    df['churn'] = (df.last_trip_date < date_cutoff).astype(int)
    return df
