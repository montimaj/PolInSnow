import gdal
import numpy as np
import os
import scipy.optimize as scp
import subprocess
import xmltodict
from glob import glob

MEAN_INC_TDX = {'12292015': (33.04875564575195 + 34.621795654296875) / 2.,
                '01082016': (38.07691192626953 + 39.37236785888672) / 2.,
                '01092016': (32.936092376708984 + 34.7651252746582) / 2.,
                '01192016': (38.038177490234375 + 39.37773895263672) / 2.,
                '01202016': (32.93116569519043 + 34.76059341430664) / 2.,
                '01302016': (38.072940826416016 + 39.38078689575195) / 2.}  # degrees

MEAN_INC_TSX = {'12292015': (33.07475662231445 + 34.60667610168457) / 2.,
                '01082016': (38.104190826416016 + 39.37824630737305) / 2.,
                '01092016': (32.974159240722656 + 34.826759338378906) / 2.,
                '01192016': (38.101966857910156 + 39.38127517700195) / 2.,
                '01202016': (32.97199821472168 + 34.81568145751953) / 2.,
                '01302016': (38.10858917236328 + 39.38400650024414) / 2.}  # degrees

ACQUISITION_ORIENTATION = {'12292015': 'ASC', '01082016': 'DESC', '01092016': 'ASC', '01192016': 'DESC',
                           '01202016': 'ASC', '01302016': 'DESC'}
VERTICAL_WAVENUMBER_SCALE_FACTOR = {'ASC': 3, 'DESC': 5}
WAVELENGTH = 3.10880853  # cm
HOA = {'12292015': 1854, '01082016': 6318, '01092016': 1761, '01192016': 6334, '01202016': 1753, '01302016': 6202}  # cm
# BPERP = 9634  # cm
NO_DATA_VALUE = -32768
STANDING_SNOW_DENSITY = {'12292015': 0.382, '01082016': 0.315, '01092016': 0.304, '01192016': 0.347, '01202016': 0.338,
                         '01302016': 0.210}  # g/cm^3
STANDING_SNOW_DEPTH = {'12292015': 36.70, '01082016': 54.90, '01092016': 56.00, '01192016': 42.80, '01202016': 42.80,
                       '01302016': 70.00}  # cm
DHUNDI_COORDS = (700089.771, 3581794.5556)  # UTM 43N


def read_images(image_path, common_path, imgformat='*.tif'):
    """
    Read images in a directory
    :param image_path: Directory path to date specific polarization files
    :param common_path: Directory path to common files
    :param imgformat: Type of image file
    :return: Dictionary of GDAL Opened references/ pointers to specific files
    """

    print("Reading images...")
    images = {}
    image_files = os.path.join(image_path, imgformat)
    common_files = os.path.join(common_path, imgformat)
    os_sep = image_path.rfind(os.sep)
    if os_sep == -1:
        os_sep = image_path.rfind('/')
    image_date = image_path[os_sep + 1:]
    layover_file = os.path.join(common_path, ACQUISITION_ORIENTATION[image_date] + os.sep + imgformat)
    file_list = glob(image_files) + glob(common_files) + glob(layover_file)
    for file in file_list:
        print(file)
        os_sep = file.rfind(os.sep)
        if os_sep == -1:
            os_sep = file.rfind('/')
        key = file[os_sep + 1: file.rfind('.')]
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


def calc_interferogram(image_dict, pol_vec, outdir, outfile, apply_masks=True, verbose=True, wf=True, load_files=False):
    """
    Calculate Pol-InSAR interferogram
    :param image_dict: Image dictionary containing GDAL references
    :param pol_vec: Polarisation vector
    :param outdir: Output directory
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
            np.save(os.path.join(outdir, 'S1_' + outfile), s1)
            np.save(os.path.join(outdir, 'S2_' + outfile), s2)
            np.save(os.path.join(outdir, 'Ifg__' + outfile), ifg)
            write_file(ifg.copy(), hv_file, os.path.join(outdir, 'Ifg__Polinsar_' + outfile))
    else:
        outfile += '.npy'
        s1, s2, ifg = np.load(os.path.join(outdir, 'S1_' + outfile)), np.load(os.path.join(outdir, 'S2_' + outfile)), \
                      np.load(os.path.join(outdir, 'Ifg__' + outfile))
    return s1, s2, ifg


def get_interferogram(image_dict, outdir):
    """
    Read topographic phase removed interferogram. The preferred option is to use #calc_interferogram(...)
    :param image_dict: Image dictionary containing GDAL references
    :param outdir: Output directory
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

    np.save(os.path.join(outdir, 'S1'), s1)
    np.save(os.path.join(outdir, 'S2'), s2)
    np.save(os.path.join(outdir, 'Ifg'), ifg)

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


def calc_coherence_mat(s1, s2, ifg, img_dict, outdir, outfile, num_looks=10, apply_masks=True, verbose=True, wf=True):
    """
    Calculate complex coherency matrix based on looks
    :param s1: Master image array
    :param s2: Slave image array
    :param ifg: Interferogram array
    :param img_dict: Image dictionary containing GDAL references
    :param outdir: Output directory
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
        outfile = os.path.join(outdir, 'Coherence_' + outfile)
        np.save(outfile, tmat)
        write_file(np.abs(tmat.copy()), img_dict['LIA'], outfile, is_complex=False)
    return tmat


def calc_ensemble_cohmat(s1, s2, ifg, img_dict, outdir, outfile, wsize=(5, 5), apply_masks=True, verbose=True,
                         wf=False):
    """
    Calculate complex coherency matrix based on ensemble averaging. This is the preferred way of calculating coherence.
    :param s1: Master image array
    :param s2: Slave image array
    :param ifg: Interferogram array
    :param img_dict: Image dictionary containing GDAL references
    :param outdir: Output directory
    :param outfile: Output file path
    :param wsize: Ensemble window size (should be half of desired window size)
    :param apply_masks: Set true for applying layover and forest masks
    :param verbose: Set true for detailed logs
    :param wf: Set true to save intermediate results
    :return: Nan fixed complex coherency matrix
    """

    lia_file = img_dict['LIA']
    num = get_ensemble_avg(ifg, wsize=wsize, image_file=lia_file, outfile='Num', is_complex=True, verbose=verbose,
                           wf=False, outdir=outdir)
    d1 = get_ensemble_avg((s1 * np.conj(s1)).real, wsize=wsize, image_file=lia_file, outfile='D1', verbose=verbose,
                          wf=False, outdir=outdir)
    d2 = get_ensemble_avg((s2 * np.conj(s2)).real, wsize=wsize, image_file=lia_file, outfile='D2', verbose=verbose,
                          wf=False, outdir=outdir)

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
        outfile = os.path.join(outdir, 'Coherence_Ensemble_' + outfile)
        np.save(outfile, tmat)
        write_file(np.abs(tmat.copy()), lia_file, outfile, is_complex=False)
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


def get_ensemble_avg(image_arr, wsize, image_file, outdir, outfile, stat='mean', scale_factor=None,
                     verbose=True, wf=False, is_complex=False, load_file=False):
    """
    Perform Ensemble Filtering based on mean, median or maximum
    :param image_arr: Image array to filter
    :param wsize: Ensemble window size (should be half of desired window size)
    :param image_file: Original GDAL reference for writing output image
    :param outdir: Output directory
    :param outfile: Outfile file path
    :param stat: Statistics to use while ensemble filtering (mean, med, max)
    :param scale_factor: Scale factor to apply (specifically used for vertical wavenumber)
    :param verbose: Set true for detailed logs
    :param wf: Set true to save intermediate results
    :param is_complex: Set true for complex values
    :param load_file: Set true to load an earlier npy file
    :return: Ensemble filtered array
    """

    if not load_file:
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
            outfile = os.path.join(outdir, outfile)
            np.save(outfile, emat)
            write_file(emat.copy(), image_file, outfile, is_complex=is_complex)
        return emat
    return np.load(os.path.join(outdir, outfile + '.npy'))


def get_ground_phase(tmat_vol, tmat_surf, wsize, outdir, img_dict, apply_masks, verbose=True, wf=True, load_file=False):
    """
    Calculate ground phase for HH-VV polarisation vector
    :param tmat_vol: Volume coherence array
    :param tmat_surf: Surface coherence array
    :param wsize: Ensemble window size (should be half of desired window size)
    :param outdir: Output directory
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
                                verbose=verbose, wf=wf, outdir=outdir)
    return np.load(os.path.join(outdir, 'Ground_Med.npy'))


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
    sinc_inv_approx = 1 - 2 * np.arcsin(val ** 0.8) / np.pi
    try:
        sinc_inv = scp.newton(mysinc, args=(val,), x0=sinc_inv_approx)
        return sinc_inv
    except RuntimeError:
        print('Root error ', val)
    return sinc_inv_approx


def calc_snow_depth_hybrid(tmat_vol, ground_phase, kz, img_file, outdir, eta=0.4, coherence_threshold=0.5, max_sd=100,
                           wf=False, load_file=False):
    """
    Calculate snow depth using Pol-InSAR based hybrid height inversion model
    :param tmat_vol: Volume coherence array
    :param ground_phase: Ground phase array
    :param kz: Vertical wavenumber array
    :param img_file: Original GDAL reference for writing output image
    :param outdir: Output directory
    :param eta: Snow depth scaling factor (0<=eta<=1)
    :param coherence_threshold: Coherence threshold (0<=coherence_threshold<=1)
    :param max_sd: Maximum snow depth (cm) in the study area (X-band won't penetrate beyond 100 cm)
    :param wf: Set true to save intermediate results
    :param load_file: Set true to load existing numpy binary and skip computation
    :return: Snow depth array
    """

    if not load_file:
        k1 = np.arctan2(tmat_vol.imag, tmat_vol.real) % (2 * np.pi) - ground_phase
        abs_tvol = np.abs(tmat_vol)
        k2 = eta * calc_sinc_inv(abs_tvol)
        kv = np.abs(k1 + k2)
        snow_depth = kv / kz
        snow_depth[abs_tvol < coherence_threshold] = 0
        snow_depth %= max_sd
        if wf:
            np.save(os.path.join(outdir, 'KV'), kv)
            np.save(os.path.join(outdir, 'Snow_Depth'), snow_depth)
            np.save(os.path.join(outdir, 'Wavenumber'), kz)
            write_file(snow_depth.copy(), img_file, os.path.join(outdir, 'Snow_Depth_Polinsar'), is_complex=False)
        return snow_depth
    return np.load(os.path.join(outdir, 'Snow_Depth.npy'))


def get_total_swe(ssd_arr, density, img_file, outdir, wf=True):
    """
    Calculate total snow water equivalent (SWE) in mm or kg/m^3
    :param ssd_arr: Standing snow depth array in cm
    :param density: Snow density (scalar or array) in g/cm^3
    :param img_file: Original image file containing affine transformation parameters
    :param outdir: Output directory
    :param wf: Set true to write intermediate files
    :return: SWE array
    """

    swe = ssd_arr * density * 10
    if wf:
        outfile = os.path.join(outdir, 'SSD_SWE')
        np.save(outfile, swe)
        write_file(swe.copy(), img_file, outfile=outfile, is_complex=False)
    return swe


def make_gdal_sys_call_str(gdal_path, gdal_command, args, verbose=False):
    """
    Make GDAL system call string
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param gdal_command: GDAL command to use
    :param args: GDAL arguments as a list
    :param verbose: Set True to print system call info
    :return: GDAL system call string,
    """

    sys_call = [gdal_path + gdal_command] + args
    if os.name == 'nt':
        gdal_path += 'OSGeo4W.bat'
        sys_call = [gdal_path] + [gdal_command] + args
    if verbose:
        print(sys_call)
    return sys_call


def retrieve_pixel_coords(geo_coord, data_source, gdal_path='/usr/bin/', verbose=False):
    """
    Get pixels coordinates from geo-coordinates
    :param geo_coord: Geo-cooridnate tuple
    :param data_source: Original GDAL reference having affine transformation parameters
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :return: Pixel coordinates in x and y direction (should be reversed in the caller function to get the actual pixel
    :param verbose: Set True to print system call info
    position)
    """

    args = ['-xml', '-geoloc', data_source, str(geo_coord[0]), str(geo_coord[1])]
    sys_call = make_gdal_sys_call_str(gdal_path=gdal_path, gdal_command='gdallocationinfo', args=args, verbose=verbose)
    p = subprocess.Popen(sys_call, stdout=subprocess.PIPE)
    p.wait()
    gdalloc_xml = xmltodict.parse(p.stdout.read())
    px, py = int(gdalloc_xml['Report']['@pixel']), int(gdalloc_xml['Report']['@line'])
    return px, py


def check_values(img_arr, img_file, geocoord, nsize=(1, 1), is_complex=False, full_stat=False):
    """
    Validate results
    :param img_arr: Image array to validate
    :param img_file: Original GDAL reference having affine transformation parameters
    :param geocoord: Geo-coordinates in tuple format
    :param nsize: Validation window size (should be half of the desired window size)
    :param is_complex: Set true for complex images such as the coherency image
    :param full_stat: Return min, max, mean and standard deviation if true, mean and sd if false
    :return: Tuple containing statistics
    """

    px, py = retrieve_pixel_coords(geocoord, gdal_path=GDAL_PATH, data_source=img_file.GetDescription())
    if is_complex:
        img_arr = np.abs(img_arr)
    img_loc = get_ensemble_window(img_arr, (py, px), nsize)
    mean = np.nanmean(img_loc)
    sd = np.nanstd(img_loc)
    if full_stat:
        return np.nanmin(img_loc), np.nanmax(img_loc), mean, sd
    return mean, sd


def get_coherence(s1, s2, ifg, wsize, img_dict, apply_masks, coh_type, verbose, wf, outdir, outfile, validate=False,
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
    :param outdir: Output directory
    :param outfile: Output file path
    :param validate: Validate results if set to true
    :param load_file: Set true to load existing numpy binary and skip computation
    :return: Complex coherency matrix and window size string as tuple
    """

    cr = list()
    if coh_type == 'E':
        ws1, ws2 = wsize[0] // 2, wsize[1] // 2
        wstr = '(' + str(wsize[0]) + ',' + str(wsize[1]) + ')'
        print('Computing Coherence mat for ' + wstr + '...')
        if not load_file:
            tmat = calc_ensemble_cohmat(s1, s2, ifg, apply_masks=apply_masks, outdir=outdir, outfile=outfile,
                                        img_dict=img_dict, wsize=(ws1, ws2), verbose=verbose, wf=wf)
        else:
            tmat = np.load(os.path.join(outdir, 'Coherence_Ensemble_' + outfile + '.npy'))
        if validate:
            cr = check_values(tmat, img_dict['LIA'], geocoord=DHUNDI_COORDS, is_complex=True)
    else:
        wstr = str(wsize)
        print('Computing Coherence mat for ' + wstr + '...')
        if not load_file:
            tmat = calc_coherence_mat(s1, s2, ifg, outdir=outdir, outfile=outfile, apply_masks=apply_masks,
                                      img_dict=img_dict, num_looks=wsize, verbose=verbose, wf=wf)
        else:
            tmat = np.load(os.path.join(outdir, 'Coherence_' + outfile + '.npy'))
        if validate:
            cr = check_values(tmat, img_dict['LIA'], geocoord=DHUNDI_COORDS, is_complex=True)
    if validate:
        cr_str = wstr + ' ' + ' '.join([str(r) for r in cr]) + '\n'
        print(cr_str)
    return tmat, wstr


def compute_vertical_wavenumber(lia_file, image_date, is_single_pass=True, verbose=True):
    """
    Calculate vertical wavenumber
    :param lia_file: Local incidence angle GDAL reference
    :param image_date: Image acquisition date string (mmddyyyy) for extracting correct incidence angles
    :param is_single_pass: Set true for single-pass acquisitions
    :param verbose: Set true for detailed logs
    :return: Vertical wavenumber array

    """
    lia = get_image_array(lia_file)
    del_theta = np.abs(MEAN_INC_TDX[image_date] - MEAN_INC_TSX[image_date])
    m = 2
    if is_single_pass:
        m = 1
    kz = np.abs(m * 2 * np.pi * np.deg2rad(del_theta) / (WAVELENGTH * np.sin(np.deg2rad(lia))))
    if verbose:
        print('Mean kz:', np.nanmean(kz), 'Min kz:', np.nanmin(kz), 'Max kz:', np.nanmax(kz))
    return kz


def senstivity_analysis(image_dict, outdir, cw, coh_type='L', image_date='12292015', apply_masks=True, lf_ifg=False,
                        lf_other=False):
    """
    Main caller function for sensitivity analysis
    :param image_dict: Image dictionary containing GDAL references
    :param outdir: Output directory to store intermediate files
    :param cw: Coherence window for ensemble averaging (coh_type should be 'L')
    :param coh_type: Set 'L' for look based coherence and 'E' for ensemble window based
    :image_date: Image date string (mmddyyyy) for selecting appropriate snow density
    :param apply_masks: Set true for applying layover and forest masks
    :param lf_ifg: Load existing files related to the PolInSAR interferogram
    :param lf_other: Load other existing files such as coherence, ground phase, etc.
    :return: None
    """

    pol_vec = calc_pol_vec_dict()
    print('Calculating s1, s2 and ifg ...')
    ifg_dir = os.path.join(outdir, 'Common')
    makedirs([ifg_dir])
    s1_vol, s2_vol, ifg_vol = calc_interferogram(image_dict, pol_vec['HV'], apply_masks=apply_masks,
                                                 outfile='Vol', verbose=False, load_files=lf_ifg, outdir=ifg_dir)
    s1_surf, s2_surf, ifg_surf = calc_interferogram(image_dict, pol_vec['HH-VV'], apply_masks=apply_masks,
                                                    outfile='Surf', verbose=False, load_files=lf_ifg, outdir=ifg_dir)
    print('Creating senstivity parameters ...')
    clooks = [3]
    cwindows = {'E': cw.copy(), 'L': clooks}
    eta_values = [0.65]
    coherence_threshold = [0.]
    cval = True
    wf = True
    lia_file = image_dict['LIA']

    outfile = open('SSD_Results_New.csv', 'a+')
    if not outfile.read():
        outfile.write('Date CWindow Epsilon CThreshold Mean_SSD(cm) SD_SSD(cm) Mean_SWE(mm) SD_SWE(mm)\n')
    print('Computation started...')
    for wsize in cwindows[coh_type]:
        output_dir = 'C' + coh_type + '_' + str(wsize)
        if isinstance(wsize, tuple):
            w1 = wsize[0]
            w2 = wsize[1]
            wsize_gp_kz = w1 // 2, w2 // 2  # window size of ground phase and kz, actual size is halved.
            output_dir = 'C' + coh_type + '_' + str(w1)
            if w1 != w2:
                output_dir += '_' + str(w2)
        elif isinstance(wsize, int):
            wsize_gp_kz = wsize // 2, wsize // 2
        output_dir = os.path.join(outdir, output_dir)
        makedirs([output_dir])
        tmat_vol, wstr = get_coherence(s1_vol, s2_vol, ifg_vol, outfile='Vol', wsize=wsize, coh_type=coh_type,
                                       apply_masks=apply_masks, img_dict=image_dict, verbose=False, wf=wf,
                                       validate=cval, load_file=lf_other, outdir=output_dir)
        tmat_surf, wstr = get_coherence(s1_surf, s2_surf, ifg_surf, outfile='Surf', wsize=wsize, coh_type=coh_type,
                                        apply_masks=apply_masks, img_dict=image_dict, verbose=False, wf=wf,
                                        validate=cval, load_file=lf_other, outdir=output_dir)
        print('Computing ground phase ...')

        ground_phase = get_ground_phase(tmat_vol, tmat_surf, wsize_gp_kz, img_dict=image_dict, apply_masks=apply_masks,
                                        verbose=False, wf=wf, load_file=lf_other, outdir=output_dir)
        print('Computing vertical wavenumber ...')
        kz = compute_vertical_wavenumber(lia_file, verbose=True, image_date=image_date)
        wstr = str(wsize)
        for eta in eta_values:
            for ct in coherence_threshold:
                print('Computing snow depth ...')
                snow_depth = calc_snow_depth_hybrid(tmat_vol, ground_phase, kz, img_file=lia_file, eta=eta,
                                                    coherence_threshold=ct, wf=wf, load_file=False, outdir=output_dir)
                snow_depth = get_ensemble_avg(snow_depth, outfile='SD_Avg', wsize=wsize, image_file=image_dict['LIA'],
                                              verbose=False, wf=wf, outdir=outdir)
                swe = get_total_swe(snow_depth, density=STANDING_SNOW_DENSITY[image_date], img_file=lia_file,
                                    outdir=output_dir)
                vr = check_values(snow_depth, lia_file, DHUNDI_COORDS)
                vr_str = ' '.join([str(r) for r in vr])
                vr = check_values(swe, lia_file, DHUNDI_COORDS)
                vr_str2 = ' '.join([str(r) for r in vr])
                final_str = image_date + ' ' + wstr + ' ' + str(eta) + ' ' + str(ct) + ' ' + vr_str + ' ' + vr_str2 + '\n'
                print(final_str)
                outfile.write(final_str)
    outfile.close()


def makedirs(directory_list):
    """
    Create directory for storing files
    :param directory_list: List of directories to create
    :return: None
    """

    for directory_name in directory_list:
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)


image_dates = ACQUISITION_ORIENTATION.keys()
completed = []
GDAL_PATH = 'C:/OSGeo4W64/'
for image_date in image_dates:
    if image_date not in completed:
        print('Working with', image_date, 'data...\n')
        base_path = 'Project_Data'
        image_path = os.path.join(base_path, image_date)
        common_path = os.path.join(base_path, 'Common')
        output_path = os.path.join('Outputs', image_date)
        image_dict = read_images(image_path=image_path, common_path=common_path)
        print('Images loaded...\n')
        cw = [(35, 35), (5, 5), (15, 15), (25, 25), (45, 45), (55, 55), (57, 57), (65, 65)]
        senstivity_analysis(image_dict, cw=cw, coh_type='E', image_date=image_date, outdir=output_path, lf_ifg=True,
                            lf_other=True)
