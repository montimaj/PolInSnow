import pandas as pd
import numpy as np
import os
from sklearn import metrics


def calculate_metrics(actual, est):
    """
    Calculate error metrics
    :param actual: List or dataframe of actual values
    :param est: List or dataframe of estimated values
    :return: R^2, MAE, and RMSE
    """

    mae = np.round(metrics.mean_absolute_error(actual, est), 2)
    r_squared = np.round(metrics.r2_score(actual, est), 2)
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

    stat_df = stat_df.__deepcopy__()
    stat_df['SSD_Error'] = np.abs(stat_df['SSD_Actual(cm)'] - stat_df['Mean_SSD_Est(cm)'])
    t = stat_df[stat_df.SSD_Error == np.min(stat_df.SSD_Error)]
    t.to_csv('Analysis_Results/Foo.csv', index=False, sep=';', mode='a')
    return np.float(stat_df[stat_df.SSD_Error == np.min(stat_df.SSD_Error)].SF), \
           np.float(stat_df[stat_df.SSD_Error == np.min(stat_df.SSD_Error)].Eta)


def generate_error_metrics(input_csv, result_file, fixed_window=None, fix_date='12292015'):
    """
    Generate RMSE, R2, and MAE for SSD and SSWE based on acquisition orientation
    :param input_csv: Input csv
    :param result_file: Output file to store error statistics
    :param fixed_window: Specify a window size for which error metrics are to be calculated
    :param fix_date: Fix scaling factor for this date
    :return: Error dataframe
    """

    stat_df = pd.read_csv(input_csv, sep=';')
    fix_date = pd.to_datetime(fix_date, format='%m%d%Y').date()
    stat_df.loc[stat_df.Date == str(fix_date), 'Mean_SSD_Est(cm)'] /= 10
    stat_df.loc[stat_df.Date == str(fix_date), 'Mean_SSWE_Est(mm)'] /= 10
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
    error_df.to_csv(result_file, index=False, sep=';')
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


def generate_error_metrics2(input_csv, result_file, fixed_window=None, fix_date='12292015'):
    """
    Generate RMSE, R2, and MAE for SSD and SSWE based on each image
    :param input_csv: Input csv
    :param result_file: Output file to store error statistics
    :param fixed_window: Specify a window size for which error metrics are to be calculated
    :param fix_date: Fix scaling factor for this date
    :return: Error dataframe
    """

    stat_df = pd.read_csv(input_csv, sep=';')
    fix_date = pd.to_datetime(fix_date, format='%m%d%Y').date()
    stat_df.loc[stat_df.Date == str(fix_date), 'Mean_SSD_Est(cm)'] /= 10
    stat_df.loc[stat_df.Date == str(fix_date), 'Mean_SSWE_Est(mm)'] /= 10
    date_list, window_list = list(set(stat_df.Date)), list(set(stat_df.CWindow))
    date_list.sort()
    window_list.sort()
    if fixed_window:
        window_list = [fixed_window]
    error_df = pd.DataFrame()
    for image_date in date_list:
        for window in window_list:
            df = stat_df[(stat_df.Date == image_date) & (stat_df.CWindow == window)]
            if not df.empty:
                best_sf, eta = get_best_scale(df)
                error_df = error_df.append(df[(df.SF == best_sf) & (df.Eta == eta)])
    error_df.to_csv(result_file, index=False, sep=';')
    r2_ssd, mae_ssd, rmse_ssd = calculate_metrics(error_df['SSD_Actual(cm)'], error_df["Mean_SSD_Est(cm)"])
    r2_sswe, mae_sswe, rmse_sswe = calculate_metrics(error_df['SSWE_Actual(mm)'], error_df["Mean_SSWE_Est(mm)"])
    print(r2_ssd, mae_ssd, rmse_ssd)
    print(r2_sswe, mae_sswe, rmse_sswe)
    return error_df


input_dir = 'Analysis_Results'
stat_csv = os.path.join(input_dir, 'Sensitivity_Results_T2.csv')
result_csv = os.path.join(input_dir, 'Error_Analysis.csv')
error_stat_df = generate_error_metrics2(stat_csv, result_file=result_csv, fixed_window='(5, 5)')
