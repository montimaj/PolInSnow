from sklearn import metrics
import pandas as pd
import numpy as np


def error_metrics(actual, est):
    """
    Calculate error metrics
    :param actual: List or dataframe of actual values
    :param est: List or dataframe of estimated values
    :return: R^2, MAE, and RMSE
    """

    mae = np.round(metrics.mean_absolute_error(actual, est), 2)
    r_squared = np.round(metrics.r2_score(actual, est), 2)
    rmse = np.round(np.sqrt(metrics.mean_squared_error(actual, est)), 2)
    return r_squared, mae, rmse


ssd_df = pd.read_csv('SSD_Results.csv', sep=' ')
subset_ssd_df = ssd_df[['Date', 'Mean_SSD(cm)', 'Actual_SSD(cm)', 'Mean_SSWE(mm)', 'Actual_SSWE(mm)']].round(2)
print(subset_ssd_df)
actual_ssd = ssd_df['Actual_SSD(cm)']
est_ssd = ssd_df['Mean_SSD(cm)']
error_ssd = np.round(actual_ssd - est_ssd, 2)

actual_swe = ssd_df['Actual_SSWE(mm)']
est_swe = ssd_df['Mean_SSWE(mm)']
error_swe = np.round(actual_swe - est_swe, 2)

new_df = subset_ssd_df[['Date', 'Mean_SSD(cm)', 'Mean_SSWE(mm)']]
new_df['Error_SSD(cm)'] = error_ssd
new_df['Error_SSWE(mm)'] = error_swe

print(new_df)
print(error_metrics(actual_ssd, est_ssd))
print(error_metrics(actual_swe, est_swe))

