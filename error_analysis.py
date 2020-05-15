import pandas as pd
import numpy as np
from sklearn import metrics


def calculate_metrics(actual, est):
    """
    Calculate error metrics
    :param actual: List or dataframe of actual values
    :param est: List or dataframe of estimated values
    :return: R^2, MAE, and RMSE
    """

    mae = np.round(metrics.mean_absolute_error(actual, est), 2)
    r_squared = np.round(np.corrcoef(actual, est)[0, 1] ** 2, 2)
    rmse = np.round(metrics.mean_squared_error(actual, est, squared=False), 2)
    return r_squared, mae, rmse


def get_error_df(stat_df, acquisition_type, window, sf):
    """
    Get error dataframe
    :param stat_df: Statistics dataframe
    :param acquisition_type: Acquisition type
    :param window: Window size
    :param sf: Scale factor
    :return: Error dataframe
    """

    actual_sd = stat_df['SSD_Actual(cm)']
    pred_sd = stat_df['Mean_SSD_Est(cm)']
    r2_sd, mae_sd, rmse_sd = calculate_metrics(actual_sd, pred_sd)
    actual_swe = stat_df['SSWE_Actual(mm)']
    pred_swe = stat_df['Mean_SSWE_Est(mm)']
    r2_swe, mae_swe, rmse_swe = calculate_metrics(actual_swe, pred_swe)
    error_dict = {'Pass': [acquisition_type], 'CWindow': [window], 'SF': [sf], 'RMSE_SSD': [rmse_sd],
                  'RMSE_SSWE': [rmse_swe], 'R2_SSD': [r2_sd], 'R2_SSWE': [r2_swe], 'MAE_SSD': [mae_sd],
                  'MAE_SSWE': [mae_swe]}
    return pd.DataFrame(data=error_dict)


def generate_error_metrics(input_csv, fixed_window=None, fix_date='12292015'):
    """
    Generate RMSE, R2, and MAE for SSD and SSWE
    :param input_csv: Input csv
    :param fixed_window: Specify a window size for which error metrics are to be calculated
    :param fix_date: Fix scaling factor for this date
    :return: None
    """

    stat_df = pd.read_csv(input_csv, sep=';')
    fix_date = pd.to_datetime(fix_date, format='%m%d%Y').date()
    stat_df.loc[(stat_df.Date == str(fix_date)) & (stat_df.SF % 10 != 0), 'SF'] = np.nan
    stat_df = stat_df.dropna(axis=0)
    stat_df.loc[(stat_df.Date == str(fix_date)) & (stat_df.SF % 10 == 0), 'SF'] /= 10
    pass_list, window_list, sf_list = list(set(stat_df.Pass)), list(set(stat_df.CWindow)), list(set(stat_df.SF))
    pass_list.sort()
    window_list.sort()
    sf_list.sort()
    if fixed_window:
        window_list = [fixed_window]
    error_df = pd.DataFrame()
    for acquisition_type in pass_list:
        for window in window_list:
            for sf in sf_list:
                df = stat_df[(stat_df.Pass == acquisition_type) & (stat_df.CWindow == window) & (stat_df.SF == sf)]
                error_df = error_df.append(get_error_df(df, acquisition_type, window, sf))
    for window in window_list:
        for sf in sf_list:
            df = stat_df[(stat_df.CWindow == window) & (stat_df.SF == sf)]
            error_df = error_df.append(get_error_df(df, acquisition_type='All', window=window, sf=sf))
    error_df.to_csv('Error_Analysis.csv', index=False, sep=';')
    pass_list.append('All')
    for acquisition_type in pass_list:
        error_report_df = error_df[error_df.Pass == acquisition_type]
        print('Best Prediction based on MAE...')
        print(error_report_df[error_report_df.MAE_SSD == np.min(error_report_df.MAE_SSD)])
    return error_df


stat_csv = 'Sensitivity_Results_T1.csv'
error_stat_df = generate_error_metrics(stat_csv)

