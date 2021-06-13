from scipy.stats import t, median_abs_deviation
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def check_outliers(data, title):
    limit = np.std(data) * 3
    mean = np.mean(data)
    border_1 = mean - limit
    border_2 = mean + limit
    plt.style.use('ggplot')
    plt.figure(figsize=(6,4))
    plt.hist(data, bins=60)
    plt.axvline(border_1, c='b')
    plt.axvline(border_2, c='b')
    plt.title(title, y=1.01, fontsize=14)
    plt.show()
    print('mean : ', mean)
    print('border_1 : ', border_1)
    print('border_2 : ', border_2)

def grubbs_test(data, title):
    print('Grubbs test for', title)
    n = len(data)
    mean = np.mean(data)
    st_dev = np.std(data)
    numerator = max(abs(data - mean))
    g_calculated = numerator / st_dev
    print('Grubbs Calculated Value:', g_calculated)
    t_value = t.ppf(1 - 0.05 / (2 * n), n - 2)
    a = ((n - 1) * np.sqrt(np.square(t_value)))
    b = (np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value)))
    g_critical = a / b
    print('Grubbs Critical Value:', g_critical)
    if g_critical > g_calculated:
        print('Calculated value < critical value, there is no outliers\n')
    else:
        print('Calculated value > critical value, there is an outlier\n')

def z_score(data, title):
    mean = np.mean(data)
    st_dev = np.std(data)
    out = []
    for i in data: 
        z = (i - mean) / st_dev
        if np.abs(z) > 3: 
            out.append(i)
    print(f'Outliers for {title}: {out}')

def robust_z_score(data, title):
    med = np.median(data)
    med_abs_dev = median_abs_deviation(data)
    out = []
    for i in data: 
        z = (0.6745 * (i - med)) / (np.median(med_abs_dev))
        if np.abs(z) > 3:
            out.append(i)
    print(f'Outliers for {title}: {out}')

def iqr_outliers(data, title):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_tail = q1 - 1.5 * iqr
    upper_tail = q3 + 1.5 * iqr
    out = []
    for i in data:
        if i > upper_tail or i < lower_tail:
            out.append(i)
    print(f'Outliers for {title}: {out}')

def winsorization(data, title):
    q1 = np.percentile(data, 1)
    q3 = np.percentile(data, 99)
    out = []
    for i in data:
        if i > q3 or i < q1:
            out.append(i)
    print(f'Outliers for {title}: {out}')

def dbscan_outliers(data):
    outlier_detection = DBSCAN(eps=2, metric='euclidean', min_samples=5)
    clusters = outlier_detection.fit_predict(data.values.reshape(-1,1))
    df = pd.DataFrame()
    df['cluster'] = clusters
    print(df['cluster'].value_counts().sort_values(ascending=False))

def iso_forest(data):
    iso = IsolationForest(random_state=1)
    preds = iso.fit_predict(data.values.reshape(-1,1))
    df = pd.DataFrame()
    df['cluster'] = preds
    print(df['cluster'].value_counts().sort_values(ascending=False))
