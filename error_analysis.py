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
    :param acquisition_type: Image acquisition type
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


def get_best_scale(stat_df):
    """
    Get best SF for a fixed window
    :param stat_df: Statistics dataframe
    :return: Best SF
    """

    stat_df["SSD_Error"] = np.abs(stat_df['SSD_Actual(cm)'] - stat_df['Mean_SSD_Est(cm)'])
    return np.float(stat_df[stat_df.SSD_Error == np.min(stat_df.SSD_Error)].SF)


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
    best_stat_df = pd.DataFrame()
    print('Best Prediction based on MAE...')
    for acquisition_type in pass_list:
        error_report_df = error_df[error_df.Pass == acquisition_type]
        best_stat_df = best_stat_df.append(error_report_df[error_report_df.MAE_SSD == np.min(error_report_df.MAE_SSD)])
    print(best_stat_df)
    actual_ssd = []
    actual_sswe = []
    pred_ssd = []
    pred_sswe = []
    for acquisition_type, window, sf in zip(best_stat_df['Pass'], best_stat_df['CWindow'], best_stat_df['SF']):
        best_stat = stat_df[(stat_df.Pass == acquisition_type) & (stat_df.CWindow == window) & (stat_df.SF == sf)]
        actual_ssd.append(list(best_stat['SSD_Actual(cm)']))
        pred_ssd.append(list(best_stat['Mean_SSD_Est(cm)']))
        actual_sswe.append(list(best_stat['SSWE_Actual(mm)']))
        pred_sswe.append(list(best_stat['Mean_SSWE_Est(mm)']))
    actual_ssd = [item for sublist in actual_ssd for item in sublist]
    pred_ssd = [item for sublist in pred_ssd for item in sublist]
    actual_sswe = [item for sublist in actual_sswe for item in sublist]
    pred_sswe = [item for sublist in pred_sswe for item in sublist]
    print('Actual SSD:', actual_ssd)
    print('Pred_SSD:', pred_ssd)
    print('Actual_SSWE:', actual_sswe)
    print('Pred_SSWE:', pred_sswe)
    print('SSD Metrics:', '(R2,', 'MAE,', 'RMSE)', calculate_metrics(actual_ssd, pred_ssd))
    print('SSWE Metrics:', '(R2,', 'MAE,', 'RMSE)', calculate_metrics(actual_sswe, pred_sswe))
    return error_df


stat_csv = 'Sensitivity_Results_T5.csv'
error_stat_df = generate_error_metrics(stat_csv, fixed_window='(5, 5)')

