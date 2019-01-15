import gdal
import numpy as np
import glob
import os
import scipy.optimize as scp

MEAN_INC_TDX = (38.07691192626953 + 39.37236785888672) / 2.
MEAN_INC_TSX = (38.104190826416016 + 39.37824630737305) / 2.
DEL_THETA = np.abs(MEAN_INC_TDX - MEAN_INC_TSX)
WAVELENGTH = 3.10880853
NO_DATA_VALUE = -32768


def read_images(path, imgformat='*.tif'):
    print("Reading images...")
    images = {}
    files = os.path.join(path, imgformat)
    for file in glob.glob(files):
        key = file[file.rfind('/') + 1: file.rfind('.')]
        images[key] = gdal.Open(file)
    print("Finished reading")
    return images


def get_complex_image(img_file):
    mst = img_file.GetRasterBand(1).ReadAsArray() + img_file.GetRasterBand(2).ReadAsArray() * 1j
    slv = img_file.GetRasterBand(3).ReadAsArray() + img_file.GetRasterBand(4).ReadAsArray() * 1j
    mst[mst.real == NO_DATA_VALUE] = np.nan
    slv[slv.real == NO_DATA_VALUE] = np.nan
    return mst, slv


def get_image_array(img_file):
    arr = img_file.GetRasterBand(1).ReadAsArray()
    arr[arr == NO_DATA_VALUE] = np.nan
    return arr


def calc_pol_vec(alpha, beta, eps, mu):
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    mu = np.deg2rad(mu)
    return np.round(np.array([[np.cos(alpha), np.sin(alpha) * np.cos(beta * np.exp(eps * 1j)),
                               np.sin(alpha) * np.sin(beta * np.exp(mu * 1j))]]), 10)


def calc_pol_vec_dict():
    pol_vec_dict = dict()
    pol_vec_dict['HH'] = calc_pol_vec(45, 0, 0, 0)
    pol_vec_dict['HV'] = pol_vec_dict['VH'] = calc_pol_vec(90, 90, 0, 0)
    pol_vec_dict['VV'] = calc_pol_vec(45, 180, 0, 0)
    pol_vec_dict['HH+VV'] = calc_pol_vec(0, 0, 0, 0)
    pol_vec_dict['HH-VV'] = calc_pol_vec(90, 0, 0, 0)
    pol_vec_dict['LL'] = calc_pol_vec(90, 45, 0, 90)
    pol_vec_dict['LR'] = pol_vec_dict['HH+VV']
    pol_vec_dict['RR'] = calc_pol_vec(90, 45, 0, -90)
    return pol_vec_dict


def calc_interferogram(image_dict, pol_vec):
    hh_file = image_dict['HH']
    hv_file = image_dict['HV']
    vh_file = image_dict['VH']
    vv_file = image_dict['VV']
    lia_file = image_dict['LIA']
    fe_file = image_dict['FE']
    topo_file = image_dict['TOPO']

    hh_mst, hh_slv = get_complex_image(hh_file)
    hv_mst, hv_slv = get_complex_image(hv_file)
    vh_mst, vh_slv = get_complex_image(vh_file)
    vv_mst, vv_slv = get_complex_image(vv_file)
    hv_mst, hv_slv = (hv_mst + vh_mst) / 2., (hv_slv + vh_slv) / 2.

    lia = get_image_array(lia_file)
    fe = get_image_array(fe_file)
    topo = get_image_array(topo_file)

    ifg = np.full_like(hh_mst, np.nan, dtype=np.complex)
    s1 = np.full_like(hh_mst, np.nan, dtype=np.complex)
    s2 = np.full_like(hh_mst, np.nan, dtype=np.complex)

    for itr in np.ndenumerate(ifg):
        idx = itr[0]
        hh_1 = hh_mst[idx]
        vv_1 = vv_mst[idx]
        hh_2 = hh_slv[idx]
        vv_2 = vv_slv[idx]
        hv_1 = hv_mst[idx]
        hv_2 = hv_slv[idx]

        k1 = (2 ** -0.5) * np.array([[hh_1 + vv_1, hh_1 - vv_1, 2 * hv_1]])
        k2 = (2 ** -0.5) * np.array([[hh_2 ** 2 + vv_2 ** 2, hh_2 ** 2 - vv_2 ** 2, 2 * hv_2 ** 2]])
        s1[idx] = np.matmul(pol_vec, k1.T)[0][0]
        s2[idx] = np.matmul(pol_vec, k2.T)[0][0]
        ifg[idx] = s1[idx] * np.conj(s2[idx]) * np.exp(fe[idx] * -1j)
        print('At ', idx, ' IFG = ', ifg[idx])

    kz = 4 * np.pi * DEL_THETA / (WAVELENGTH * np.sin(lia))

    np.save('S1', s1)
    np.save('S2', s2)
    np.save('Ifg', ifg)
    np.save('kz', kz)

    write_file(ifg, hh_file, 'Ifg_Polinsar')
    write_file(s1, hh_file, 'S1')
    write_file(s2, hh_file, 'S2')
    write_file(kz, hh_file, 'Wavenumber')

    return s1, s2, ifg, kz, topo


def write_file(arr, src_file, outfile='test', no_data_value=NO_DATA_VALUE, is_complex=True):
    arr[np.isnan(arr.real)] = NO_DATA_VALUE
    driver = gdal.GetDriverByName("ENVI")
    if is_complex:
        out = driver.Create(outfile, arr.shape[1], arr.shape[0], 1, gdal.GDT_CFloat32)
    else:
        out = driver.Create(outfile, arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    out.SetProjection(src_file.GetProjection())
    out.SetGeoTransform(src_file.GetGeoTransform())
    out.GetRasterBand(1).SetNoDataValue(no_data_value)
    out.GetRasterBand(1).WriteArray(arr)
    out.FlushCache()


def calc_coherence_mat(s1, s2, ifg, img_file, num_looks=10):
    tmat = np.full_like(ifg, np.nan, dtype=np.complex)
    max_x = ifg.shape[0]
    for itr in np.ndenumerate(tmat):
        idx = itr[0]
        start_x = idx[0]
        end_x = start_x + num_looks
        if end_x > max_x:
            end_x = max_x
        sub_s1 = s1[start_x: end_x]
        sub_s2 = s2[start_x: end_x]
        num = np.sum(ifg[start_x: end_x])
        denom = np.sqrt(np.sum(sub_s1 * np.conj(sub_s1))) * np.sqrt(np.sum(sub_s2 * np.conj(sub_s2)))
        tmat[idx] = num / denom
        print('Coherence at ', idx, '= ', np.abs(tmat[idx]))
    np.save('Coherence', tmat)
    write_file(tmat, img_file, 'Coherence')
    return tmat


def mysinc(x, c):
    return np.sinc(x) - c


def calc_snow_depth(tmat, kz, topo, img_file, eps=0.4):
    abs_tmat = np.abs(tmat)
    sinc_inv = scp.newton(mysinc, args=(abs_tmat, ), x0=1)
    snow_depth = (np.arctan(tmat.imag / tmat.real) + 2 * eps * sinc_inv - topo) / kz
    np.save('Snow_Depth', snow_depth)
    write_file(snow_depth, img_file, 'Snow_Depth_Polinsar', is_complex=False)
    return snow_depth


image_dict = read_images('../THESIS/SnowSAR/Polinsar/Clipped_Tifs')
print('Images loaded...\n')
pol_vec_HV = calc_pol_vec_dict()['HV']
s1, s2, ifg, kz, topo = calc_interferogram(image_dict, pol_vec_HV)
print('Starting coherence matrix calculation ...')
tmat = calc_coherence_mat(s1, s2, ifg, img_file=image_dict['HV'])
print('Calculating snow depth')
snow_depth = calc_snow_depth(tmat, kz, topo, img_file=image_dict['HV'])
