import gdal
import numpy as np
import glob
import os
import scipy.optimize as scp
import affine

MEAN_INC_TDX = (38.07691192626953 + 39.37236785888672) / 2.
MEAN_INC_TSX = (38.104190826416016 + 39.37824630737305) / 2.
WAVELENGTH = 3.10880853
HOA = 6318
BPERP = 9634
R = 98420.04252609884
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


def get_complex_image(img_file, is_dual=False):
    mst = img_file.GetRasterBand(1).ReadAsArray() + img_file.GetRasterBand(2).ReadAsArray() * 1j
    mst[mst == np.complex(NO_DATA_VALUE, NO_DATA_VALUE)] = np.nan
    if not is_dual:
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


def calc_interferogram(image_dict, pol_vec, outfile, verbose=True, wf=True):
    hh_file = image_dict['HH']
    hv_file = image_dict['HV']
    vh_file = image_dict['VH']
    vv_file = image_dict['VV']
    fe_file = image_dict['FE']

    hh_mst, hh_slv = get_complex_image(hh_file)
    hv_mst, hv_slv = get_complex_image(hv_file)
    vh_mst, vh_slv = get_complex_image(vh_file)
    vv_mst, vv_slv = get_complex_image(vv_file)
    hv_mst, hv_slv = (hv_mst + vh_mst) / 2., (hv_slv + vh_slv) / 2.

    fe = get_image_array(fe_file)

    ifg = np.full_like(hv_mst, np.nan, dtype=np.complex)
    s1 = np.full_like(hv_mst, np.nan, dtype=np.complex)
    s2 = np.full_like(hv_mst, np.nan, dtype=np.complex)

    for itr in np.ndenumerate(ifg):
        idx = itr[0]
        hh_1 = hh_mst[idx]
        vv_1 = vv_mst[idx]
        hh_2 = hh_slv[idx]
        vv_2 = vv_slv[idx]
        hv_1 = hv_mst[idx]
        hv_2 = hv_slv[idx]

        nan_check = np.isnan(np.array([[hv_1, hv_2]]))
        if len(nan_check[nan_check]) == 0:
            k1 = (2 ** -0.5) * np.array([[hh_1 + vv_1, hh_1 - vv_1, 2 * hv_1]])
            k2 = (2 ** -0.5) * np.array([[hh_2 + vv_2, hh_2 - vv_2, 2 * hv_2]])
            s1[idx] = np.matmul(pol_vec, k1.T)[0][0]
            s2[idx] = np.matmul(pol_vec, k2.T)[0][0]
            ifg[idx] = s1[idx] * np.conj(s2[idx]) * np.exp(fe[idx] * -1j)
            if verbose:
                print('At ', idx, ' IFG = ', ifg[idx])
    if wf:
        np.save('Out/S1_' + outfile, s1)
        np.save('Out/S2_' + outfile, s2)
        np.save('Out/Ifg_' + outfile, ifg)
        # write_file(ifg, hv_file, 'Out/Ifg_Polinsar')
        # write_file(s1, hh_file, 'Out/S1')
        # write_file(s2, hh_file, 'Out/S2')

    return s1, s2, ifg


def get_interferogram(image_dict):
    hv_file = image_dict['HV']
    vh_file = image_dict['VH']
    ifg = get_complex_image(image_dict['IFG'], is_dual=True)

    hv_mst, hv_slv = get_complex_image(hv_file)
    vh_mst, vh_slv = get_complex_image(vh_file)
    hv_mst, hv_slv = (hv_mst + vh_mst) / 2., (hv_slv + vh_slv) / 2.

    s1 = (2 ** 0.5) * hv_mst
    s2 = (2 ** 0.5) * hv_slv

    np.save('Out/S1', s1)
    np.save('Out/S2', s2)
    np.save('Out/Ifg', ifg)

    return s1, s2, ifg


def nanfix_tmat_val(tmat, idx, verbose=True):
    i = 1
    while True:
        window = get_ensemble_window(tmat, idx, (i, i))
        tval = np.nanmean(window)
        if verbose:
            print('\nTVAL nanfix', i, np.abs(tval))
        if not np.isnan(tval):
            return tval
        i += 1


def nanfix_tmat_arr(tmat_arr, lia_arr, verbose=True):
    for idx, tval in np.ndenumerate(tmat_arr):
        if not np.isnan(lia_arr[idx]):
            if np.isnan(tval):
                tmat_arr[idx] = nanfix_tmat_val(tmat_arr, idx, verbose)
    return tmat_arr


def calc_coherence_mat(s1, s2, ifg, lia_file, img_file, outfile, num_looks=10, verbose=True, wf=True):
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
            if verbose:
                print('Coherence at ', idx, '= ', np.abs(tmat[idx]))
    lia_arr = get_image_array(lia_file)
    tmat = nanfix_tmat_arr(tmat, lia_arr)
    if wf:
        np.save('Out/Coherence_' + outfile, tmat)
        write_file(tmat.copy(), img_file, 'Out/Coherence_' + outfile)
    return tmat


def calc_ensemble_cohmat(s1, s2, ifg, lia_file, img_file, outfile, wsize=(5, 5), verbose=True, wf=False):
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
    lia_arr = get_image_array(lia_file)
    tmat = nanfix_tmat_arr(tmat, lia_arr)
    if wf:
        np.save('Out/Coherence_Ensemble_' + outfile, tmat)
        write_file(tmat.copy(), img_file, 'Out/Coherence_Ensemble_' + outfile)
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


def get_ensemble_avg(image_arr, wsize, image_file, outfile, stat='mean', scale=False, scale_factor=None,
                     verbose=True, wf=False):
    emat = np.full_like(image_arr, np.nan, dtype=np.float32)
    for index, value in np.ndenumerate(image_arr):
        if not np.isnan(value):
            ensemble_window = get_ensemble_window(image_arr, index, wsize)
            if stat == 'mean':
                emat[index] = np.nanmean(ensemble_window)
            elif stat == 'med':
                emat[index] = np.nanmedian(ensemble_window)
            elif stat == 'max':
                emat[index] = np.nanmax(ensemble_window)
            if scale:
                emat[index] *= scale_factor
            if verbose:
                print(index, emat[index])
    if wf:
        outfile = 'Out/' + outfile
        np.save(outfile, emat)
    write_file(emat.copy(), image_file, outfile, is_complex=False)
    return emat


def get_ground_phase(tmat_vol, tmat_surf, wsize, image_file):
    a = np.abs(tmat_surf) ** 2 - 1
    b = 2 * np.real((tmat_vol - tmat_surf) * np.conj(tmat_surf))
    c = np.abs(tmat_vol - tmat_surf) ** 2
    lws = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    lws[lws > 1] = 1
    lws[lws < 0] = 0
    t = tmat_vol - tmat_surf * (1 - lws)
    ground_phase = np.arctan2(t.imag, t.real) % (2 * np.pi)
    return get_ensemble_avg(ground_phase, outfile='Ground_Med', wsize=wsize, image_file=image_file, stat='med',
                            wf=True)


def calc_snow_depth_hybrid(tmat_vol, ground_phase, kz, lia_file, eps=0.4, coherence_threshold=0.5, verbose=True, wf=False):
    lia = get_image_array(lia_file)
    snow_depth = np.full_like(tmat_vol, np.nan, dtype=np.float32)
    for itr in np.ndenumerate(snow_depth):
        idx = itr[0]
        lia_val = np.deg2rad(lia[idx])
        tval_vol = tmat_vol[idx]
        gval = ground_phase[idx]
        kz_val = kz[idx]
        if not np.isnan(lia_val):
            abs_tval_vol = np.abs(tval_vol)
            snow_depth[idx] = 0
            if abs_tval_vol >= coherence_threshold:
                t1 = np.arctan2(tval_vol.imag, tval_vol.real) % (2 * np.pi)
                k1 = t1 - gval
                k2 = 2 * eps * scp.newton(mysinc, args=(abs_tval_vol,), x0=1)
                kv = k1 + k2
                snow_depth[idx] = np.abs(kv / kz_val)
                if verbose:
                    print('At ', idx, '(kz, t1, t2, k1, k2, kv)= ', kz_val, t1, gval, k1, k2, kv, 'Snow depth= ', snow_depth[idx])
    if wf:
        np.save('Out/Snow_Depth', snow_depth)
        np.save('Out/Wavenumber', kz)
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
    return np.nanmin(image_arr), np.nanmax(image_arr), np.nanmean(image_arr), np.nanstd(image_arr)


def check_values(img_file, geocoords, nsize=(1, 1), is_complex=False, is_dual=True):
    img_file = gdal.Open(img_file)
    px, py = retrieve_pixel_coords(geocoords, img_file)
    if is_complex:
        img_arr = get_complex_image(img_file, is_dual=is_dual)
        img_arr = np.abs(img_arr)
    else:
        img_arr = get_image_array(img_file)
    img_loc = get_ensemble_window(img_arr, (py, px), nsize)
    return get_image_stats(img_loc)


def get_coherence(s1, s2, ifg, wsize, lia_file, img_file, coh_type, verbose, wf, outfile, validate=False):
    cr = list()
    if coh_type == 'E':
        ws1, ws2 = int(wsize[0] / 2.), int(wsize[1] / 2.)
        wstr = '(' + str(wsize[0]) + ',' + str(wsize[1]) + ')'
        print('Computing Coherence mat for ' + wstr + '...')
        tmat = calc_ensemble_cohmat(s1, s2, ifg, outfile=outfile, lia_file=lia_file, img_file=img_file,
                                    wsize=(ws1, ws2), verbose=verbose, wf=wf)
        if validate:
            cr = check_values('Out/Coherence_Ensemble_' + outfile + '.tif', geocoords=(700089.771, 3581794.5556),
                              is_complex=True)
    else:
        wstr = str(wsize)
        print('Computing Coherence mat for ' + wstr + '...')
        tmat = calc_coherence_mat(s1, s2, ifg, outfile=outfile, lia_file=lia_file, img_file=img_file, num_looks=wsize,
                                  verbose=verbose, wf=wf)
        if validate:
            cr = check_values('Out/Coherence_'+outfile + '.tif', geocoords=(700089.771, 3581794.5556), is_complex=True)
    if validate:
        cr_str = wstr + ' ' + ' '.join([str(r) for r in cr]) + '\n'
        print(cr_str)
    return tmat, wstr


def compute_vertical_wavenumber(lia_file, wsize, outfile, img_file, scale_factor, is_single_pass=True):
    lia = get_image_array(lia_file)
    # del_theta = np.deg2rad(np.abs(MEAN_INC_TDX - MEAN_INC_TSX))
    del_theta = np.abs(MEAN_INC_TDX - MEAN_INC_TSX)
    m = 4
    if is_single_pass:
        m = 2
    kz = m * np.pi * np.deg2rad(del_theta) / (WAVELENGTH * np.sin(np.deg2rad(lia)))
    kz = get_ensemble_avg(kz, wsize=wsize, image_file=img_file, scale=True, scale_factor=scale_factor, outfile=outfile,
                          wf=True)
    return kz

def senstivity_analysis(image_dict, coh_type='L'):
    pol_vec = calc_pol_vec_dict()
    print('Calculating s1, s2 and ifg ...')
    # s1_vol, s2_vol, ifg_vol = calc_interferogram(image_dict, pol_vec['HV'], outfile='Vol', verbose=False)
    # s1_surf, s2_surf, ifg_surf = calc_interferogram(image_dict, pol_vec['HH-VV'], outfile='Surf', verbose=False)
    s1_vol, s2_vol, ifg_vol = np.load('Out/S1_Vol.npy'), np.load('Out/S2_Vol.npy'), np.load('Out/Ifg_Vol.npy')
    s1_surf, s2_surf, ifg_surf = np.load('Out/S1_Surf.npy'), np.load('Out/S2_Surf.npy'), np.load('Out/Ifg_Surf.npy')
    print('Creating senstivity parameters ...')
    wrange = range(3, 66, 2)
    # ewindows = [(i, j) for i, j in zip(wrange, wrange)]
    # epsilon = np.round(np.linspace(0, 1, 11), 1)
    # clooks = range(2, 21)
    # coherence_threshold = np.round(np.linspace(0.10, 0.90, 17), 2)
    # cwindows = [(5, 5)]
    ewindows = [(65, 65)]
    clooks = [3]
    cwindows = {'E': ewindows.copy(), 'L': clooks.copy()}
    epsilon = [0.4]
    coherence_threshold = [0]
    cval = True

    outfile = open('sensitivity_ssd_new.csv', 'a+')
    outfile.write('CWindow Epsilon CThreshold SWindow Min(cm) Max(cm) Mean(cm) SD(cm)\n')
    img_file = image_dict['HV']
    lia_file = image_dict['LIA']
    print('Computation started...')
    for wsize1 in cwindows[coh_type]:
        # tmat_vol, wstr1 = get_coherence(s1_vol, s2_vol, ifg_vol, outfile='Vol', wsize=wsize1, coh_type=coh_type,
        #                                 lia_file=lia_file, img_file=img_file, verbose=False, wf=True, validate=cval)
        # tmat_surf, wstr1 = get_coherence(s1_surf, s2_surf, ifg_surf, outfile='Surf', wsize=wsize1, coh_type=coh_type,
        #                                  lia_file=lia_file, img_file=img_file, verbose=False, wf=True, validate=cval)
        tmat_vol = np.load('Out/Coherence_Vol.npy')
        # tmat_surf = np.load('Out/Coherence_Surf.npy')
        print('Computing ground phase ...')
        # ground_phase = get_ground_phase(tmat_vol, tmat_surf, (10, 10), image_file=img_file)
        ground_phase = np.load('Out/Ground_Med.npy')
        wstr1 = str(wsize1)
        for eps in epsilon:
            for ct in coherence_threshold:
                print('Computing vertical wavenumber ...')
                kz = compute_vertical_wavenumber(lia_file, img_file=img_file, scale_factor=10, outfile='Wavenumber',
                                                 wsize=(10, 10))
                # kz = np.load('Out/Wavenumber.npy')
                print('Computing snow depth ...')
                # topo = get_image_array(image_dict['TOPO']) % (2 * np.pi)
                snow_depth = calc_snow_depth_hybrid(tmat_vol, ground_phase, kz, lia_file=lia_file, eps=eps,
                                                    coherence_threshold=ct, verbose=True)
                for wsize2 in ewindows:
                    ws1, ws2 = int(wsize2[0] / 2.), int(wsize2[1] / 2.)
                    print('Ensemble averaging snow depth ...')
                    avg_sd = get_ensemble_avg(snow_depth, (ws1, ws2), img_file, outfile='Avg_SD', verbose=True)
                    vr = check_values('Avg_SD.tif', (700089.771, 3581794.5556))  # Dhundi
                    vr_str = ' '.join([str(r) for r in vr])
                    wstr2 = '(' + str(wsize2[0]) + ',' + str(wsize2[1]) + ')'
                    final_str = wstr1 + ' ' + str(eps) + ' ' + str(ct) + ' ' + wstr2 + ' ' + vr_str + '\n'
                    print(final_str)
                    outfile.write(final_str)
                    # vr = validate_dry_snow('Avg_SD.tif', (705849.1335, 3577999.4174)) # Kothi
    outfile.close()


image_dict = read_images('../THESIS/Thesis_Files/Polinsar/Clipped_Tifs')
print('Images loaded...\n')
senstivity_analysis(image_dict)