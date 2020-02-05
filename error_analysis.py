from sklearn import metrics
import pandas as pd
import numpy as np

ssd_df = pd.read_csv('SSD_Results.csv', sep=' ')
subset_ssd_df = ssd_df[['Date', 'Scale', 'CThreshold', 'Mean_SSD(cm)', 'Actual_SSD(cm)']].round(2)
print(subset_ssd_df)
actual_ssd = ssd_df['Actual_SSD(cm)']
est_ssd = ssd_df['Mean_SSD(cm)']
mae = np.round(metrics.mean_absolute_error(actual_ssd, est_ssd), 2)
r_squared = np.round(metrics.r2_score(actual_ssd, est_ssd), 2)
rmse = np.round(np.sqrt(metrics.mean_squared_error(actual_ssd, est_ssd)), 2)

print('MAE:', mae, 'R^2:', r_squared, 'RMSE:', rmse)
