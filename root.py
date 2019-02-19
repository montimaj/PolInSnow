import numpy as np
import scipy.optimize as scp
import pandas as pd


def mysinc1(x, arg):
    return np.sinc(x) - arg
	

def mysinc2(x, arg):
    return np.sin(x)/x - arg
	

def approx_sinc_inv1(x):
	return 1 - 2 * np.arcsin(x ** 0.8) / np.pi
	
def approx_sinc_inv2(x):
	return np.pi - 2 * np.arcsin(x ** 0.8)

tests = np.round(np.linspace(0.1, 0.9, 9), 1)
sinc1 = []
sinc2 = []
sinc1_inv = []
sinc2_inv = []
sinc3_inv = []
sinc4_inv = []
for t in tests:
	val1 = np.sinc(t)
	sinc1.append(np.round(val1, 3))
	val2 = np.sin(t)/t
	sinc2.append(np.round(val2, 3))
	x1 = approx_sinc_inv1(val1)
	sinc1_inv.append(np.round(scp.newton(mysinc1, args=(val1,), x0=x1), 3))
	sinc3_inv.append(np.round(x1, 3))
	x2 = approx_sinc_inv2(val2)
	sinc4_inv.append(np.round(x2, 3))
	sinc2_inv.append(np.round(scp.newton(mysinc2, args=(val2,), x0=x2), 3))
df = pd.DataFrame({'test_val':tests, 'sinc1': sinc1, 'sinc2': sinc2, 'sinc1_inv': sinc1_inv, 'sinc2_inv': sinc2_inv, 'sinc1_ainv': sinc3_inv, 'sinc2_ainv': sinc4_inv})
df.to_csv(r'D:\SnowSAR\Snow_Analysis\SSD_Sensitivity\root_results.csv', index=False)