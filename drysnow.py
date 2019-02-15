import gdal
import numpy as np
import glob
import os
import scipy.optimize as scp
import affine

MEAN_INC_TDX = (38.07691192626953 + 39.37236785888672) / 2.
MEAN_INC_TSX = (38.104190826416016 + 39.37824630737305) / 2.
WAVELENGTH = 3.10880853  # cm
HOA = 6318  # cm
BPERP = 9634 # cm
NO_DATA_VALUE = -32768
STANDING_SNOW_DENSITY = 0.315  # g/cm^3
DHUNDI_COORDS = (700089.771, 3581794.5556)  # UTM 43N


def read_images(path, imgformat='*.tif'):
    """
    Read images in a directory
    :param path: Directory path
    :param imgformat: Type of image file
    :return: Dictionary of GDAL Opened references/ pointers to specific files
    """

    print("Reading images...")
    images = {}
    files = os.path.join(path, imgformat)
    for file in glob.glob(files):
        key = file[file.rfind('/') + 1: file.rfind('.')]
        images[key] = gdal.Open(file)
    print("Finished reading")
    return images


def get_complex_image(img_file, is_dual=False):
    """
    Read complex image stored either in four bands or two separate bands and set nan values accordingly
    :param img_file: GDAL reference file
    :param is_dual: True if image file is stored in two separate bands
    :return: If dual is set true, a complex numpy array is returned, numpy array tuple otherwise
    """

    mst = img_file.GetRasterBand(1).ReadAsArray() + img_file.GetRasterBand(2).ReadAsArray() * 1j
    mst[mst == np.complex(NO_DATA_VALUE, NO_DATA_VALUE)] = np.nan
    if not is_dual:
        slv = img_file.GetRasterBand(3).ReadAsArray() + img_file.GetRasterBand(4).ReadAsArray() * 1j
        slv[slv == np.complex(NO_DATA_VALUE, NO_DATA_VALUE)] = np.nan
        return mst, slv
    return mst


def get_image_array(img_file):
    """
    Read real numpy arrays from file
    :param img_file: GDAL reference file
    :return: Numpy array with nan set accordingly
    """

    arr = img_file.GetRasterBand(1).ReadAsArray()
    arr[arr == NO_DATA_VALUE] = np.nan
    return arr


def set_nan_img(img_arr, layover_file, forest_file):
    """
    Set nan values to specific images using layover and forest masks
    :param img_arr: Image array whose nan values are to be set
    :param layover_file: Layover file, no layover marked with 0
    :param forest_file: Forest file, forest marked with 0
    :return: Nan set array
    """

    layover = get_image_array(layover_file)
    forest = get_image_array(forest_file)
    for idx, lval in np.ndenumerate(layover):
        if np.round(lval) != 0 or forest[idx] == 0:
            img_arr[idx] = np.nan
    return img_arr


def calc_pol_vec(alpha, beta, eps, mu):
    """
    Calculate Pauli polarisation vector
    :param alpha: alpha value
    :param beta: beta value
    :param eps: epsilon value
    :param mu: mu value
    :return: Polarisation vectors
    """

    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    mu = np.deg2rad(mu)
    return np.round(np.array([[np.cos(alpha), np.sin(alpha) * np.cos(beta * np.exp(eps * 1j)),
                               np.sin(alpha) * np.sin(beta * np.exp(mu * 1j))]]), 10)


def calc_pol_vec_dict():
    """
    Generate polarisation vector dictionary
    :return: Dictionary of polarisation vectors
    """

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
    """
    Write image files in TIF format
    :param arr: Image array to write
    :param src_file: Original image file for retrieving affine transformation parameters
    :param outfile: Output file path
    :param no_data_value: No data value to be set
    :param is_complex: If true, write complex image array in two separate bands
    :return: None
    """

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


def calc_interferogram(image_dict, pol_vec, outfile, apply_masks=True, verbose=True, wf=True, load_files=False):
    """
    Calculate Pol-InSAR interferogram
    :param image_dict: Image dictionary containing GDAL references
    :param pol_vec: Polarisation vector
    :param outfile: Output file(s) path
    :param apply_masks: Set true for applying layover and shadow masks
    :param verbose: Set true for detailed logs
    :param wf: Set true for writing intermediate results
    :param load_files: Set true to load existing numpy binaries and skip computation
    :return: Tuple containing master array, slave array and interferogram
    """

    if not load_files:
        hh_file = image_dict['HH']
        hv_file = image_dict['HV']
        vh_file = image_dict['VH']
        vv_file = image_dict['VV']
        fe_file = image_dict['FE']
        layover_file = image_dict['LAYOVER']
        forest_file = image_dict['FOREST']

        hh_mst, hh_slv = get_complex_image(hh_file)
        hv_mst, hv_slv = get_complex_image(hv_file)
        vh_mst, vh_slv = get_complex_image(vh_file)
        vv_mst, vv_slv = get_complex_image(vv_file)
        hv_mst, hv_slv = (hv_mst + vh_mst) / 2., (hv_slv + vh_slv) / 2.

        fe = get_image_array(fe_file) % (2 * np.pi)

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
        if apply_masks:
            s1 = set_nan_img(s1, layover_file, forest_file)
            s2 = set_nan_img(s2, layover_file, forest_file)
            ifg = set_nan_img(ifg, layover_file, forest_file)
        if wf:
            np.save('Out/S1_' + outfile, s1)
            np.save('Out/S2_' + outfile, s2)
            np.save('Out/Ifg_' + outfile, ifg)
            write_file(ifg.copy(), hv_file, 'Out/Ifg_Polinsar_' + outfile)
            # write_file(s1, hh_file, 'Out/S1_' + outfile)
            # write_file(s2, hh_file, 'Out/S2_' + outfile)
    else:
        s1, s2, ifg = np.load('Out/S1_' + outfile + '.npy'), np.load('Out/S2_' + outfile + '.npy'), \
                      np.load('Out/Ifg_' + outfile + '.npy')
    return s1, s2, ifg


def get_interferogram(image_dict):
    """
    Read topographic phase removed interferogram. The preferred option is to use #calc_interferogram(...)
    :param image_dict: Image dictionary containing GDAL references
    :return: Tuple containing master array, slave array and interferogram
    """

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
    """
    Fix nan value occuring due to incorrect terrain correction
    :param tmat: Complex coherency matrix
    :param idx: Index at which the nan value is to be replaced by the mean of its neighbourhood
    :param verbose: Set true for detailed logs
    :return: Corrected element
    """

    i = 1
    while True:
        window = get_ensemble_window(tmat, idx, (i, i))
        tval = np.nanmean(window)
        if verbose:
            print('\nTVAL nanfix', i, np.abs(tval))
        if not np.isnan(tval):
            return tval
        i += 1


def nanfix_tmat_arr(tmat_arr, lia_arr, layover_arr=None, forest_arr=None, apply_masks=True, verbose=False):
    """
    Fix nan values occuring due to incorrect terrain correction
    :param tmat_arr: Complex coherency matrix
    :param lia_arr: Local incidence angle array or any coregistered image array having non-nan values
    in the area of interest
    :param layover_arr: Layover array, no layover marked with 0
    :param forest_arr: Forest array, forest marked with 0
    :param apply_masks: Set true for applying layover and forest masks
    :param verbose: Set true for detailed logs
    :return:
    """

    for idx, tval in np.ndenumerate(tmat_arr):
        if not np.isnan(lia_arr[idx]):
            if np.isnan(tval):
                if apply_masks:
                    if np.round(layover_arr[idx]) == 0 and forest_arr[idx] == 1:
                        tmat_arr[idx] = nanfix_tmat_val(tmat_arr, idx, verbose)
                else:
                    tmat_arr[idx] = nanfix_tmat_val(tmat_arr, idx, verbose)
    return tmat_arr


def calc_coherence_mat(s1, s2, ifg, img_dict, outfile, num_looks=10, apply_masks=True, verbose=True, wf=True):
    """
    Calculate complex coherency matrix based on looks
    :param s1: Master image array
    :param s2: Slave image array
    :param ifg: Interferogram array
    :param img_dict: Image dictionary containing GDAL references
    :param outfile: Output file path
    :param num_looks: Number of looks to apply
    :param apply_masks: Set true for applying layover and forest masks
    :param verbose: Set true for detailed logs
    :param wf: Set true to save intermediate results
    :return: Nan fixed complex coherency matrix
    """

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
    lia_arr = get_image_array(img_dict['LIA'])
    if apply_masks:
        layover_arr = get_image_array(img_dict['LAYOVER'])
        forest_arr = get_image_array(img_dict['FOREST'])
        tmat = nanfix_tmat_arr(tmat, lia_arr, layover_arr, forest_arr)
    else:
        tmat = nanfix_tmat_arr(tmat, lia_arr, apply_masks=False)
    if wf:
        np.save('Out/Coherence_' + outfile, tmat)
        write_file(tmat.copy(), img_dict['LIA'], 'Out/Coherence_' + outfile)
    return tmat


def calc_ensemble_cohmat(s1, s2, ifg, img_dict, outfile, wsize=(5, 5), apply_masks=True, verbose=True, wf=False):
    """
    Calculate complex coherency matrix based on ensemble averaging
    :param s1: Master image array
    :param s2: Slave image array
    :param ifg: Interferogram array
    :param img_dict: Image dictionary containing GDAL references
    :param outfile: Output file path
    :param wsize: Ensemble window size (should be half of desired window size)
    :param apply_masks: Set true for applying layover and forest masks
    :param verbose: Set true for detailed logs
    :param wf: Set true to save intermediate results
    :return: Nan fixed complex coherency matrix
    """

    lia_file = img_dict['LIA']
    num = get_ensemble_avg(ifg, wsize=wsize, image_file=lia_file, outfile='Num', is_complex=True,
                           verbose=verbose, wf=False)
    d1 = get_ensemble_avg((s1 * np.conj(s1)).real, wsize=wsize, image_file=lia_file, outfile='D1',
                          verbose=verbose, wf=False)
    d2 = get_ensemble_avg((s2 * np.conj(s2)).real, wsize=wsize, image_file=lia_file, outfile='D2',
                          verbose=verbose, wf=False)

    tmat = num / (np.sqrt(d1 * d2))
    tmat[np.abs(tmat) > 1] = 1 + 0j

    lia_arr = get_image_array(lia_file)
    if apply_masks:
        layover_arr = get_image_array(img_dict['LAYOVER'])
        forest_arr = get_image_array(img_dict['FOREST'])
        tmat = nanfix_tmat_arr(tmat, lia_arr, layover_arr, forest_arr)
    else:
        tmat = nanfix_tmat_arr(tmat, lia_arr, apply_masks=False)
    if wf:
        np.save('Out/Coherence_Ensemble_' + outfile, tmat)
        write_file(tmat.copy(), lia_file, 'Out/Coherence_Ensemble_' + outfile)
    return tmat


def get_ensemble_window(image_arr, index, wsize):
    """
    Subset image array based on the window size
    :param image_arr: Image array whose subset is to be returned
    :param index: Central subset index
    :param wsize: Ensemble window size (should be half of desired window size)
    :return: Subset array
    """

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


def get_ensemble_avg(image_arr, wsize, image_file, outfile, stat='mean', scale_factor=None,
                     verbose=True, wf=False, is_complex=False):
    """
    Perform Ensemble Filtering based on mean, median or maximum
    :param image_arr: Image array to filter
    :param wsize: Ensemble window size (should be half of desired window size)
    :param image_file: Original GDAL reference for writing output image
    :param outfile: Outfile file path
    :param stat: Statistics to use while ensemble filtering (mean, med, max)
    :param scale_factor: Scale factor to apply (specifically used for vertical wavenumber)
    :param verbose: Set true for detailed logs
    :param wf: Set true to save intermediate results
    :param is_complex: Set true for complex values
    :return: Ensemble filtered array
    """

    dt = np.float32
    if is_complex:
        dt = np.complex
    emat = np.full_like(image_arr, np.nan, dtype=dt)
    for index, value in np.ndenumerate(image_arr):
        if not np.isnan(value):
            ensemble_window = get_ensemble_window(image_arr, index, wsize)
            if stat == 'mean':
                emat[index] = np.nanmean(ensemble_window)
            elif stat == 'med':
                emat[index] = np.nanmedian(ensemble_window)
            elif stat == 'max':
                emat[index] = np.nanmax(ensemble_window)
            if scale_factor:
                emat[index] *= scale_factor
            if verbose:
                print(index, emat[index])
    if wf:
        outfile = 'Out/' + outfile
        np.save(outfile, emat)
        write_file(emat.copy(), image_file, outfile, is_complex=is_complex)
    return emat


def get_ground_phase(tmat_vol, tmat_surf, wsize, img_dict, apply_masks, verbose=True, wf=True, load_file=False):
    """
    Calculate ground phase for HH-VV polarisation vector
    :param tmat_vol: Volume coherence array
    :param tmat_surf: Surface coherence array
    :param wsize: Ensemble window size (should be half of desired window size)
    :param img_dict: Image dictionary containing GDAL references
    :param apply_masks: Set true for applying layover and forest masks
    :param verbose: Set true for detailed logs
    :param wf: Set true to write intermediate files
    :param load_file: Set true to load existing numpy binary and skip computation
    :return: Median filtered ground phase
    """

    if not load_file:
        a = np.abs(tmat_surf) ** 2 - 1
        b = 2 * np.real((tmat_vol - tmat_surf) * np.conj(tmat_surf))
        c = np.abs(tmat_vol - tmat_surf) ** 2
        lws = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        lws[lws > 1] = 1
        lws[lws < 0] = 0
        t = tmat_vol - tmat_surf * (1 - lws)
        ground_phase = np.arctan2(t.imag, t.real) % (2 * np.pi)
        lia_arr = get_image_array(img_dict['LIA'])
        if apply_masks:
            layover_arr = get_image_array(img_dict['LAYOVER'])
            forest_arr = get_image_array(img_dict['FOREST'])
            ground_phase = nanfix_tmat_arr(ground_phase, lia_arr, layover_arr, forest_arr)
        else:
            ground_phase = nanfix_tmat_arr(ground_phase, lia_arr, apply_masks=False)
        return get_ensemble_avg(ground_phase, outfile='Ground_Med', wsize=wsize, image_file=img_dict['LIA'], stat='med',
                                verbose=verbose, wf=wf)
    return np.load('Out/Ground_Med.npy')


def mysinc(x, c):
    """
    Custom SINC function for root finding in the hybrid height inversion model
    :param x: SINC argument
    :param c: Constant
    :return: SINC(x) - c
    """

    return np.sinc(x) - c


def calc_sinc_inv(val):
    """
    Calculate sinc inverse
    :param val: Input value
    :return: sinc inverse
    """
    if val == 1:
        return 0
    sinc_inv_approx = np.pi - 2 * np.arcsin(val ** 0.8)
    try:
        sinc_inv = scp.newton(mysinc, args=(val,), x0=1)
        if not np.isnan(sinc_inv):
            return sinc_inv
    except RuntimeError:
        print('Root error ', val)
    return sinc_inv_approx


def calc_snow_depth_hybrid(tmat_vol, ground_phase, kz, img_file, eta=0.4, coherence_threshold=0.5, verbose=True,
                           wf=False, load_file=False):
    """
    Calculate snow depth using Pol-InSAR based hybrid height inversion model
    :param tmat_vol: Volume coherence array
    :param ground_phase: Ground phase array
    :param kz: Vertical wavenumber array
    :param img_file: Original GDAL reference for writing output image
    :param eta: Snow depth scaling factor (0<=eta<=1)
    :param coherence_threshold: Coherence threshold (0<=coherence_threshold<=1)
    :param verbose: Set true for detailed logs
    :param wf: Set true to save intermediate results
    :param load_file: Set true to load existing numpy binary and skip computation
    :return: Snow depth array
    """

    if not load_file:
        snow_depth = np.full_like(tmat_vol, np.nan, dtype=np.float32)
        kv = snow_depth.copy()
        for itr in np.ndenumerate(snow_depth):
            idx = itr[0]
            tval_vol = tmat_vol[idx]
            gval = ground_phase[idx]
            kz_val = kz[idx]
            if not np.isnan(tval_vol):
                abs_tval_vol = np.abs(tval_vol)
                snow_depth[idx] = 0
                if abs_tval_vol > coherence_threshold:
                    t1 = np.arctan2(tval_vol.imag, tval_vol.real) % (2 * np.pi)
                    k1 = t1 - gval
                    k2 = 0
                    if eta > 0:
                        k2 = eta * calc_sinc_inv(abs_tval_vol)
                    kv[idx] = k1 + k2
                    snow_depth[idx] = kv[idx] / kz_val
                    if snow_depth[idx] < 0:
                        snow_depth[idx] = 0
                    if verbose:
                        print('At ', idx, '(kz, t1, t2, k1, k2, kv)= ', kz_val, t1, gval, k1, k2, kv[idx],
                              'Snow depth= ', snow_depth[idx])
        if wf:
            np.save('Out/KV', kv)
            np.save('Out/Snow_Depth', snow_depth)
            np.save('Out/Wavenumber', kz)
            write_file(snow_depth.copy(), img_file, 'Snow_Depth_Polinsar', is_complex=False)
        return snow_depth
    return np.load('Out/Snow_Depth.npy')


def get_total_swe(ssd_arr, density, img_file, wf=True):
    """
    Calculate total snow water equivalent (SWE) in mm or kg/m^3
    :param ssd_arr: Standing snow depth array in cm
    :param density: Snow density (scalar or array) in g/cm^3
    :param img_file: Original image file containing affine transformation parameters
    :param wf: Set true to write intermediate files
    :return: SWE array
    """

    swe = ssd_arr * density * 10
    if wf:
        np.save('Out/SSD_SWE', swe)
        write_file(swe.copy(), img_file, outfile='Out/SSD_SWE', is_complex=False)
    return swe


def retrieve_pixel_coords(geo_coord, data_source):
    """
    Get pixels coordinates from geo-coordinates
    :param geo_coord: Geo-cooridnate tuple
    :param data_source: Original GDAL reference having affine transformation parameters
    :return: Pixel coordinates in x and y direction (should be reversed in the caller function to get the actual pixel
    position)
    """

    x, y = geo_coord[0], geo_coord[1]
    forward_transform = affine.Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    px, py = int(px + 0.5), int(py + 0.5)
    return px, py


def check_values(img_arr, img_file, geocoords, nsize=(1, 1), is_complex=False, full_stat=False):
    """
    Validate results
    :param img_arr: Image array to validate
    :param img_file: Original GDAL reference having affine transformation parameters
    :param geocoords: Geo-coordinates in tuple format
    :param nsize: Validation window size (should be half of the desired window size)
    :param is_complex: Set true for complex images such as the coherency image
    :param full_stat: Return min, max, mean and standard deviation if true, mean and sd if false
    :return: Tuple containing statistics
    """

    px, py = retrieve_pixel_coords(geocoords, img_file)
    if is_complex:
        img_arr = np.abs(img_arr)
    img_loc = get_ensemble_window(img_arr, (py, px), nsize)
    mean = np.nanmean(img_loc)
    sd = np.nanstd(img_loc)
    if full_stat:
        return np.nanmin(img_loc), np.nanmax(img_loc), mean, sd
    return mean, sd


def get_coherence(s1, s2, ifg, wsize, img_dict, apply_masks, coh_type, verbose, wf, outfile, validate=False,
                  load_file=False):
    """
    Coherency matrix caller function
    :param s1: Master image array
    :param s2: Slave image array
    :param ifg: Interferogram array
    :param wsize: Ensemble window size (should be half of desired window size) or number of looks
    :param img_dict: Image dictionary containing GDAL references
    :param apply_masks: Set true for applying layover and forest masks
    :param coh_type: Set 'L' for look based coherence and 'E' for ensemble window based
    :param verbose: Set true for detailed logs
    :param wf: Set true to save intermediate results
    :param outfile: Output file path
    :param validate: Validate results if set to true
    :param load_file: Set true to load existing numpy binary and skip computation
    :return: Complex coherency matrix and window size string as tuple
    """

    cr = list()
    if coh_type == 'E':
        ws1, ws2 = int(wsize[0] / 2.), int(wsize[1] / 2.)
        wstr = '(' + str(wsize[0]) + ',' + str(wsize[1]) + ')'
        print('Computing Coherence mat for ' + wstr + '...')
        if not load_file:
            tmat = calc_ensemble_cohmat(s1, s2, ifg, apply_masks=apply_masks, outfile=outfile, img_dict=img_dict,
                                        wsize=(ws1, ws2), verbose=verbose, wf=wf)
        else:
            tmat = np.load('Out/Coherence_Ensemble_' + outfile + '.npy')
        if validate:
            cr = check_values(tmat, img_dict['LIA'], geocoords=DHUNDI_COORDS, is_complex=True)
    else:
        wstr = str(wsize)
        print('Computing Coherence mat for ' + wstr + '...')
        if not load_file:
            tmat = calc_coherence_mat(s1, s2, ifg, outfile=outfile, apply_masks=apply_masks, img_dict=img_dict,
                                      num_looks=wsize, verbose=verbose, wf=wf)
        else:
            tmat = np.load('Out/Coherence_' + outfile + '.npy')
        if validate:
            cr = check_values(tmat, img_dict['LIA'], geocoords=DHUNDI_COORDS, is_complex=True)
    if validate:
        cr_str = wstr + ' ' + ' '.join([str(r) for r in cr]) + '\n'
        print(cr_str)
    return tmat, wstr


def compute_vertical_wavenumber(lia_file, wsize, outfile, scale_factor, is_single_pass=True, verbose=True, wf=True,
                                load_file=False):
    """
    Calculate vertical wavenumber
    :param lia_file: Local incidence angle GDAL reference
    :param wsize: Ensemble window size (should be half of desired window size)
    :param outfile: Output file path
    :param scale_factor: Vertical wavenumber scale factor (real valued, shoud be chosen according to the study area)
    :param is_single_pass: Set true for single-pass acquisitions
    :param verbose: Set true for detailed logs
    :param wf: Set true to write intermediate files
    :param load_file: Set true to load existing numpy binary and skip computation
    :return: Vertical wavenumber array

    """
    if not load_file:
        lia = get_image_array(lia_file)
        del_theta = np.abs(MEAN_INC_TDX - MEAN_INC_TSX)
        m = 4
        if is_single_pass:
            m = 2
        kz = m * np.pi * np.deg2rad(del_theta) / (WAVELENGTH * np.sin(np.deg2rad(lia)))
        kz = get_ensemble_avg(kz, wsize=wsize, image_file=lia_file, scale_factor=scale_factor, outfile=outfile,
                              wf=wf, verbose=verbose)
    else:
        kz = np.load('Out/Wavenumber.npy')
    return kz


def senstivity_analysis(image_dict, coh_type='L', apply_masks=True):
    """
    Main caller function for sensitivity analysis
    :param image_dict: Image dictionary containing GDAL references
    :param coh_type: Set 'L' for look based coherence and 'E' for ensemble window based
    :param apply_masks: Set true for applying layover and forest masks
    :return: None
    """

    pol_vec = calc_pol_vec_dict()
    lf = True
    print('Calculating s1, s2 and ifg ...')
    s1_vol, s2_vol, ifg_vol = calc_interferogram(image_dict, pol_vec['HV'], apply_masks=apply_masks,
                                                 outfile='Vol', verbose=False, load_files=lf)
    s1_surf, s2_surf, ifg_surf = calc_interferogram(image_dict, pol_vec['HH-VV'], apply_masks=apply_masks,
                                                    outfile='Surf', verbose=False, load_files=lf)
    print('Creating senstivity parameters ...')
    # wrange = range(45, 66, 2)
    # ewindows = [(i, j) for i, j in zip(wrange, wrange)]
    # clooks = range(2, 21)
    # coherence_threshold = np.round(np.linspace(0.10, 0.90, 17), 2)
    ewindows = [(47, 47)]
    cw = [(1, 3)]
    clooks = [3, 5, 6, 7, 9, 11]
    cwindows = {'E': cw.copy(), 'L': clooks}
    # eta_values = np.round(np.linspace(0.01, 0.09, 9), 2)
    eta_values = [0.65]
    coherence_threshold = [0.6]
    cval = True
    wf = False
    scale_factor = 5
    lia_file = image_dict['LIA']
    lf=False

    outfile = open('test_coh.csv', 'a+')
    outfile.write('CWindow Epsilon CThreshold SWindow Mean_SSD(cm) SD_SSD(cm) Mean_SWE(mm) SD_SWE(mm)\n')
    print('Computation started...')
    for wsize1 in cwindows[coh_type]:
        tmat_vol, wstr1 = get_coherence(s1_vol, s2_vol, ifg_vol, outfile='Vol', wsize=wsize1, coh_type=coh_type,
                                        apply_masks=apply_masks, img_dict=image_dict, verbose=False, wf=wf,
                                        validate=cval, load_file=lf)
        tmat_surf, wstr1 = get_coherence(s1_surf, s2_surf, ifg_surf, outfile='Surf', wsize=wsize1, coh_type=coh_type,
                                         apply_masks=apply_masks, img_dict=image_dict, verbose=False, wf=wf,
                                         validate=cval, load_file=lf)
        # print('Computing ground phase ...')
        # ground_phase = get_ground_phase(tmat_vol, tmat_surf, (10, 10), img_dict=image_dict, apply_masks=apply_masks,
        #                                 verbose=False, wf=wf, load_file=lf)
        # print('Computing vertical wavenumber ...')
        # kz = compute_vertical_wavenumber(lia_file, scale_factor=scale_factor, outfile='Wavenumber',
        #                                  wsize=(10, 10), verbose=False, wf=wf, load_file=lf)
        # wstr1 = str(wsize1)
        # for eta in eta_values:
        #     for ct in coherence_threshold:
        #         print('Computing snow depth ...')
        #         snow_depth = calc_snow_depth_hybrid(tmat_vol, ground_phase, kz, img_file=lia_file, eta=eta,
        #                                             coherence_threshold=ct, wf=wf, verbose=False, load_file=False)
        #         for wsize2 in ewindows:
        #             ws1, ws2 = int(wsize2[0] / 2.), int(wsize2[1] / 2.)
        #             print('Ensemble averaging snow depth ...')
        #             avg_sd = get_ensemble_avg(snow_depth, (ws1, ws2), image_file=lia_file, outfile='Avg_SD_47',
        #                                       verbose=False, wf=wf)
        #             swe = get_total_swe(avg_sd, density=STANDING_SNOW_DENSITY, img_file=lia_file)
        #             vr = check_values(avg_sd, lia_file, DHUNDI_COORDS)
        #             vr_str = ' '.join([str(r) for r in vr])
        #             wstr2 = '(' + str(wsize2[0]) + ',' + str(wsize2[1]) + ')'
        #             vr = check_values(swe, lia_file, DHUNDI_COORDS)
        #             vr_str2 = ' '.join([str(r) for r in vr])
        #             final_str = wstr1 + ' ' + str(eta) + ' ' + str(ct) + ' ' + wstr2 + ' ' + vr_str + ' ' + vr_str2 + '\n'
        #             print(final_str)
        #             outfile.write(final_str)
                    # vr = validate_dry_snow('Avg_SD.tif', (705849.1335, 3577999.4174)) # Kothi
    outfile.close()


image_dict = read_images('../THESIS/Thesis_Files/Polinsar/Clipped_Tifs')
print('Images loaded...\n')
senstivity_analysis(image_dict, coh_type='L')