import gdal
import numpy as np
import glob
import os
import scipy.optimize as scp
import affine

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


def get_complex_image(img_file, is_ifg=False):
    mst = img_file.GetRasterBand(1).ReadAsArray() + img_file.GetRasterBand(2).ReadAsArray() * 1j
    mst[mst == np.complex(NO_DATA_VALUE, NO_DATA_VALUE)] = np.nan
    if not is_ifg:
        slv = img_file.GetRasterBand(3).ReadAsArray() + img_file.GetRasterBand(4).ReadAsArray() * 1j
        slv[slv == np.complex(NO_DATA_VALUE, NO_DATA_VALUE)] = np.nan
        return mst, slv
    return mst


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


def write_file(arr, src_file, outfile='test', no_data_value=NO_DATA_VALUE, is_complex=True):

    driver = gdal.GetDriverByName("GTiff")
    if is_complex:
        out = driver.Create(outfile + ".tif", arr.shape[1], arr.shape[0], 2, gdal.GDT_Float32)
    else:
        out = driver.Create(outfile + ".tif", arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    out.SetProjection(src_file.GetProjection())
    out.SetGeoTransform(src_file.GetGeoTransform())
    out.GetRasterBand(1).SetNoDataValue(no_data_value)
    if is_complex:
        arr[np.isnan(arr)] = no_data_value + no_data_value * 1j
        out.GetRasterBand(2).SetNoDataValue(no_data_value)
        out.GetRasterBand(1).WriteArray(arr.real)
        out.GetRasterBand(2).WriteArray(arr.imag)
    else:
        arr[np.isnan(arr)] = no_data_value
        out.GetRasterBand(1).WriteArray(arr)
    out.FlushCache()


# def calc_interferogram(image_dict, pol_vec):
#     hh_file = image_dict['HH']
#     hv_file = image_dict['HV']
#     vh_file = image_dict['VH']
#     vv_file = image_dict['VV']
#     fe_file = image_dict['FE']
#
#     hh_mst, hh_slv = get_complex_image(hh_file)
#     hv_mst, hv_slv = get_complex_image(hv_file)
#     vh_mst, vh_slv = get_complex_image(vh_file)
#     vv_mst, vv_slv = get_complex_image(vv_file)
#     hv_mst, hv_slv = (hv_mst + vh_mst) / 2., (hv_slv + vh_slv) / 2.
#
#     fe = get_image_array(fe_file)
#
#     ifg = np.full_like(hv_mst, np.nan, dtype=np.complex)
#     s1 = np.full_like(hv_mst, np.nan, dtype=np.complex)
#     s2 = np.full_like(hv_mst, np.nan, dtype=np.complex)
#
#     for itr in np.ndenumerate(ifg):
#         idx = itr[0]
#         hh_1 = hh_mst[idx]
#         vv_1 = vv_mst[idx]
#         hh_2 = hh_slv[idx]
#         vv_2 = vv_slv[idx]
#         hv_1 = hv_mst[idx]
#         hv_2 = hv_slv[idx]
#
#         nan_check = np.isnan(np.array([[hv_1, hv_2]]))
#         if len(nan_check[nan_check]) == 0:
#             k1 = (2 ** -0.5) * np.array([[hh_1 + vv_1, hh_1 - vv_1, 2 * hv_1]])
#             k2 = (2 ** -0.5) * np.array([[hh_2 + vv_2, hh_2 - vv_2, 2 * hv_2]])
#             s1[idx] = np.matmul(pol_vec, k1.T)[0][0]
#             s2[idx] = np.matmul(pol_vec, k2.T)[0][0]
#             s1[idx] = (2 ** 0.5) * hv_1
#             s2[idx] = (2 ** 0.5) * hv_2
#             ifg[idx] = s1[idx] * np.conj(s2[idx]) * np.exp(fe[idx] * -1j)
#             print('At ', idx, ' IFG = ', ifg[idx])
#
#     np.save('Out/S1', s1)
#     np.save('Out/S2', s2)
#     np.save('Out/Ifg', ifg)
#
#     write_file(ifg, hv_file, 'Out/Ifg_Polinsar')
#     write_file(s1, hh_file, 'Out/S1')
#     write_file(s2, hh_file, 'Out/S2')
#
#     return s1, s2, ifg


def get_interferogram(image_dict):
    hv_file = image_dict['HV']
    vh_file = image_dict['VH']
    ifg = get_complex_image(image_dict['IFG'], is_ifg=True)

    hv_mst, hv_slv = get_complex_image(hv_file)
    vh_mst, vh_slv = get_complex_image(vh_file)
    hv_mst, hv_slv = (hv_mst + vh_mst) / 2., (hv_slv + vh_slv) / 2.

    s1 = (2 ** 0.5) * hv_mst
    s2 = (2 ** 0.5) * hv_slv

    np.save('Out/S1', s1)
    np.save('Out/S2', s2)
    np.save('Out/Ifg', ifg)

    return s1, s2, ifg


def calc_coherence_mat(s1, s2, ifg, img_file, num_looks=10):
    tmat = np.full_like(ifg, np.nan, dtype=np.complex)
    max_y = ifg.shape[1]
    for itr in np.ndenumerate(tmat):
        idx = itr[0]
        start_x = idx[0]
        start_y = idx[1]
        end_y = start_y + num_looks
        if end_y > max_y:
            end_y = max_y
        sub_s1 = s1[start_x][start_y: end_y]
        sub_s2 = s2[start_x][start_y: end_y]
        sub_ifg = ifg[start_x][start_y: end_y]
        nan_check = np.isnan(np.array([[sub_s1[0], sub_s2[0], sub_ifg[0]]]))
        if len(nan_check[nan_check]) == 0:
            num = np.nansum(sub_ifg)
            denom = np.sqrt(np.nansum(sub_s1 * np.conj(sub_s1))) * np.sqrt(np.nansum(sub_s2 * np.conj(sub_s2)))
            tmat[idx] = num / denom
            if np.abs(tmat[idx]) > 1:
                tmat[idx] = 1 + 0j
            print('Coherence at ', idx, '= ', np.abs(tmat[idx]))
    np.save('Out/Coherence', tmat)
    write_file(tmat, img_file, 'Out/Coherence')
    return tmat


def calc_ensemble_cohmat(s1, s2, ifg, img_file, wsize=(5, 5), verbose=True):
    tmat = np.full_like(ifg, np.nan, dtype=np.complex)
    for itr in np.ndenumerate(tmat):
        idx = itr[0]
        nan_check = np.isnan(np.array([[s1[idx], s2[idx], ifg[idx]]]))
        if len(nan_check[nan_check]) == 0:
            sub_s1 = get_ensemble_window(s1, idx, wsize)
            sub_s2 = get_ensemble_window(s2, idx, wsize)
            sub_ifg = get_ensemble_window(ifg, idx, wsize)
            num = np.nansum(sub_ifg)
            denom = np.sqrt(np.nansum(sub_s1 * np.conj(sub_s1))) * np.sqrt(np.nansum(sub_s2 * np.conj(sub_s2)))
            tmat[idx] = num / denom
            if np.abs(tmat[idx]) > 1:
                tmat[idx] = 1 + 0j
            if verbose:
                print('Coherence at ', idx, '= ', np.abs(tmat[idx]))
    if verbose:
        np.save('Out/Coherence_Ensemble', tmat)
        write_file(tmat.copy(), img_file, 'Out/Coherence_Ensemble')
    return tmat


def mysinc(x, c):
    return np.sinc(x) - c


def get_ensemble_window(image_arr, index, wsize):
    startx = index[0] - wsize[0]
    starty = index[1] - wsize[1]
    if startx < 0:
        startx = 0
    if starty < 0:
        starty = 0
    endx = index[0] + wsize[0] + 1
    endy = index[1] + wsize[1] + 1
    limits = image_arr.shape[0] + 1, image_arr.shape[1] + 1
    if endx > limits[0] + 1:
        endx = limits[0] + 1
    if endy > limits[1] + 1:
        endy = limits[1] + 1
    return image_arr[startx: endx, starty: endy]


def get_ensemble_avg(image_arr, wsize, image_file, outfile, verbose=True):
    print('PERFORMING ENSEMBLE AVERAGING...')
    emat = np.full_like(image_arr, np.nan, dtype=np.float32)
    for index, value in np.ndenumerate(image_arr):
        if not np.isnan(value):
            ensemble_window = get_ensemble_window(image_arr, index, wsize)
            emat[index] = np.nanmean(ensemble_window)
            if verbose:
                print(index, emat[index])
    if verbose:
        np.save('Out/SD_Emat', emat)
    write_file(emat.copy(), image_file, outfile, is_complex=False)
    return emat


def nanfix_tmat(tmat, idx, verbose=True):
    i = 1
    while True:
        window = get_ensemble_window(tmat, idx, (i, i))
        tval = np.nanmean(window)
        if verbose:
            print('\nTVAL nanfix', i, np.abs(tval))
        if not np.isnan(tval):
            return tval
        i += 1


def calc_snow_depth_hybrid(tmat, lia_file, eps=0.4, coherence_threshold=0.5, verbose=True):
    lia = get_image_array(lia_file)
    snow_depth = np.full_like(tmat, np.nan, dtype=np.float32)
    for itr in np.ndenumerate(snow_depth):
        idx = itr[0]
        tval = tmat[idx]
        lia_val = np.deg2rad(lia[idx])
        if not np.isnan(lia_val):
            if np.isnan(tval):
                tval = nanfix_tmat(tmat, idx, verbose)
            abs_tval = np.abs(tval)
            snow_depth[idx] = 0
            if abs_tval >= coherence_threshold:
                sinc_inv = scp.newton(mysinc, args=(abs_tval, ), x0=1)
                kz_val = 4 * np.pi * np.deg2rad(DEL_THETA) / (WAVELENGTH * np.sin(lia_val))
                snow_depth[idx] = np.abs((np.arctan(tval.imag / tval.real) + 2 * eps * sinc_inv) / kz_val)
                if verbose:
                    print('At ', idx, 'Snow depth= ', snow_depth[idx])
    if verbose:
        np.save('Out/Snow_Depth', snow_depth)
        write_file(snow_depth.copy(), lia_file, 'Snow_Depth_Polinsar', is_complex=False)
    return snow_depth


def retrieve_pixel_coords(geo_coord, data_source):
    x, y = geo_coord[0], geo_coord[1]
    forward_transform = affine.Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    px, py = int(px + 0.5), int(py + 0.5)
    return px, py


def get_image_stats(image_arr):
    return np.min(image_arr), np.max(image_arr), np.mean(image_arr), np.std(image_arr)


def validate_dry_snow(dsd_file, geocoords, nsize=(1, 1)):
    dsd_file = gdal.Open(dsd_file)
    px, py = retrieve_pixel_coords(geocoords, dsd_file)
    fsd_arr = dsd_file.GetRasterBand(1).ReadAsArray()
    fsd_dhundi = get_ensemble_window(fsd_arr, (py, px), nsize)
    return get_image_stats(fsd_dhundi[fsd_dhundi != NO_DATA_VALUE])


def senstivity_analysis(image_dict):
    print('Calculating s1, s2 and ifg ...')
    s1, s2, ifg = get_interferogram(image_dict)
    print('Creating senstivity parameters ...')
    # wrange = range(3, 66, 2)
    # cwindows = [(i, j) for i, j in zip(wrange, wrange)]
    # ewindows = cwindows.copy()
    # epsilon = np.round(np.linspace(0, 1, 11), 1)
    cwindows = [(5, 5)]
    ewindows = [(65, 65)]
    epsilon = [0.4]
    coherence_threshold = np.round(np.linspace(0.10, 0.90, 17), 2)
    outfile = open('sensitivity.csv', 'a+')
    outfile.write('CWindow Epsilon CThreshold SWindow Min(cm) Max(cm) Mean(cm) SD(cm)\n')
    img_file = image_dict['HV']
    print('Computation started...')
    for wsize1 in cwindows:
        ws1, ws2 = int(wsize1[0] / 2.), int(wsize1[1] / 2.)
        wstr1 = '(' + str(wsize1[0]) + ',' + str(wsize1[1]) + ')'
        print('Computing Coherence mat for ' + wstr1 + '...')
        tmat = calc_ensemble_cohmat(s1, s2, ifg, img_file=img_file, wsize=(ws1, ws2), verbose=False)
        for eps in epsilon:
            for ct in coherence_threshold:
                print('Computing Snow Depth ...')
                snow_depth = calc_snow_depth_hybrid(tmat, lia_file=image_dict['LIA'], eps=eps, coherence_threshold=ct, verbose=False)
                for wsize2 in ewindows:
                    ws1, ws2 = int(wsize2[0] / 2.), int(wsize2[1] / 2.)
                    print('Ensemble averaging snow depth ...')
                    avg_sd = get_ensemble_avg(snow_depth, (ws1, ws2), image_file=img_file, outfile='Avg_SD', verbose=False)
                    vr = validate_dry_snow('Avg_SD.tif', (700089.771, 3581794.5556))  # Dhundi
                    vr_str = ' '.join([str(r) for r in vr])
                    wstr2 = '(' + str(wsize2[0]) + ',' + str(wsize2[1]) + ')'
                    final_str = wstr1 + ' ' + str(eps) + ' ' + str(ct) + ' ' + wstr2 + ' ' + vr_str + '\n'
                    print(final_str)
                    outfile.write(final_str)
                    # vr = validate_dry_snow('Avg_SD.tif', (705849.1335, 3577999.4174)) # Kothi
    outfile.close()


image_dict = read_images('../THESIS/SnowSAR/Polinsar/Clipped_Tifs')
print('Images loaded...\n')
senstivity_analysis(image_dict)
#pol_vec_HV = calc_pol_vec_dict()['HV']

