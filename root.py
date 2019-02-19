import numpy as np
import scipy.optimize as scp
import pandas as pd


def mysinc1(x, arg):
    """
    Custom SINC function for root finding in the hybrid height inversion model
    :param x: SINC argument
    :param c: Constant
    :return: SINC(x) - c
    """
	
	return np.sinc(x) - arg
	

def mysinc2(x, arg):
    """
    SINC approximation function for root finding in the hybrid height inversion model
    :param x: SINC argument
	:param c: Constant
    :return: Unnormalised SINC(x) - c
    """
	
	return np.sin(x)/x - arg
	

def approx_sinc_inv(x):
	"""
    SINC approximation function for root finding in the hybrid height inversion model
    :param x: SINC argument
    :return: Approximate inverse
    """
	
	return np.pi - 2 * np.arcsin(x ** 0.8)
	
tests = np.round(np.linspace(0.1, 0.9, 10), 1)
sinc1 = []
sinc2 = []
sinc1_inv = []
sinc2_inv = []
sinc3_inv = []
sinc4_inv = []
for t in tests:
	val1 = np.round(np.sinc(t), 2)
	sinc1.append(val1)
	val2 = np.round(np.sin(t)/t, 2)
	sinc2.append(val2)
	sinc1_inv.append(np.round(scp.newton(mysinc1, args=(val1,), x0=1), 3))
	sinc2_inv.append(np.round(scp.newton(mysinc2, args=(val2,), x0=1), 3))
	sinc3_inv.append(np.round(approx_sinc_inv(val1), 3))
	sinc4_inv.append(np.round(approx_sinc_inv(val2), 3))
df = pd.DataFrame({'test_val':tests, 'sinc1': sinc1, 'sinc2': sinc2, 'sinc1_inv': sinc1_inv, 'sinc2_inv': sinc2_inv, 'sinc1_ainv': sinc3_inv, 'sinc2_ainv': sinc4_inv})
df.to_csv(r'D:\SnowSAR\Snow_Analysis\SSD_Sensitivity\root_results1.csv', index=False)