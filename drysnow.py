import gdal
import numpy as np
import os
import multiprocessing
import scipy.optimize as scp
import subprocess
import xmltodict
import pandas as pd
import warnings
from glob import glob
from joblib import Parallel, delayed

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
WAVELENGTH = 3.10880853  # cm
HOA = {'12292015': 1854, '01082016': 6318, '01092016': 1761, '01192016': 6334, '01202016': 1753, '01302016': 6202}  # cm
NO_DATA_VALUE = -32768
STANDING_SNOW_DENSITY = {'12292015': 0.382, '01082016': 0.315, '01092016': 0.304, '01192016': 0.347, '01202016': 0.338,
                         '01302016': 0.210}  # g/cm^3
STANDING_SNOW_DEPTH = {'12292015': 36.70, '01082016': 54.90, '01092016': 56.00, '01192016': 42.80, '01202016': 42.80,
                       '01302016': 70.00}  # cm
DHUNDI_COORDS = (700089.771, 3581794.5556)  # UTM 43N
GDAL_PATH = 'C:/OSGeo4W64/'


def read_images(image_path, common_path, imgformat='*.tif', verbose=False):
    """
    Read images in a directory
    :param image_path: Directory path to date specific polarization files
    :param common_path: Directory path to common files
    :param imgformat: Type of image file
    :param verbose: Set True to get extra details
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
    layover_files = os.path.join(common_path, ACQUISITION_ORIENTATION[image_date] + os.sep + imgformat)
    file_list = glob(image_files) + glob(common_files) + glob(layover_files)
    for file in file_list:
        if verbose:
            print(file)
        os_sep = file.rfind(os.sep)
        if os_sep == -1:
            os_sep = file.rfind('/')
        key = file[os_sep + 1: file.rfind('.')]
        images[key] = gdal.Open(file)
    print("Finished reading")
    return images


def get_effective_permittivity(snow_density):
    """
    Get effective pemittivity of standing snow / ice
    :param snow_density: Snow density in gm/cm^3
    :return: Real part of Effective permittivity
    """

    eff = 1 + 1.5995 * snow_density + 1.861 * (snow_density ** 3)
    return eff


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


def get_image_array(img_file, no_data_value=NO_DATA_VALUE):
    """
    Read real numpy arrays from file
    :param img_file: GDAL reference file
    :param no_data_value: No Data Value
    :return: Numpy array with nan set accordingly
    """

    arr = img_file.GetRasterBand(1).ReadAsArray()
    arr[arr == no_data_value] = np.nan
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

    print('Calculating s1, s2 and ifg ...')
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
        s1, s2 = np.load(os.path.join(outdir, 'S1_' + outfile)), np.load(os.path.join(outdir, 'S2_' + outfile))
        ifg = np.load(os.path.join(outdir, 'Ifg__' + outfile))
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

    print('Computing ground phase ...')
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


def calc_snow_depth_hybrid(tmat_vol, ground_phase, kz, img_file, outdir, eta=0.4, coherence_threshold=0.5, wf=False,
                           load_file=False, ensemble_avg=True, **kwargs):
    """
    Calculate snow depth using Pol-InSAR based hybrid height inversion model, 'wsize' must be passed via kwargs to
    calculate validation statistics
    :param tmat_vol: Volume coherence array
    :param ground_phase: Ground phase array
    :param kz: Vertical wavenumber array
    :param img_file: Original GDAL reference for writing output image
    :param outdir: Output directory
    :param eta: Snow depth scaling factor (0<=eta<=1)
    :param coherence_threshold: Coherence threshold (0<=coherence_threshold<=1)
    :param wf: Set true to save intermediate results
    :param load_file: Set true to load existing numpy binary and skip computation
    :param ensemble_avg: Set False to directly return snow depth array without ensemble averaging. If True, pass 
    appropriate wsize and verbose parameters.
    :return: Snow depth array and results over the coordinates
    """

    print('Computing snow depth ...')
    if not load_file:
        k1 = (np.arctan2(tmat_vol.imag, tmat_vol.real) % (2 * np.pi)) - ground_phase
        abs_tvol = np.abs(tmat_vol)
        k2 = eta * calc_sinc_inv(abs_tvol)
        kv = np.abs(k1 + k2)
        snow_depth = kv / kz
        snow_depth[abs_tvol < coherence_threshold] = 0
        if wf:
            np.save(os.path.join(outdir, 'KV'), kv)
            np.save(os.path.join(outdir, 'Snow_Depth'), snow_depth)
            write_file(snow_depth.copy(), img_file, os.path.join(outdir, 'Snow_Depth_Polinsar'), is_complex=False)
        if ensemble_avg:
            snow_depth = get_ensemble_avg(snow_depth, outfile='SD_Avg', wsize=kwargs['wsize'], image_file=img_file, 
                                          verbose=kwargs['verbose'], wf=wf, outdir=outdir)
    else:
        if ensemble_avg:
            snow_depth = np.load(os.path.join(outdir, 'SD_Avg.npy'))
        else:
            snow_depth = np.load(os.path.join(outdir, 'Snow_Depth.npy'))
    cr = check_values(snow_depth, img_file, nsize=kwargs['wsize'])
    return snow_depth, cr


def get_total_swe(ssd_arr, density, img_file, outdir, wsize, wf=True):
    """
    Calculate total snow water equivalent (SWE) in mm or kg/m^3
    :param ssd_arr: Standing snow depth array in cm
    :param density: Snow density (scalar or array) in g/cm^3
    :param img_file: Original image file containing affine transformation parameters
    :param outdir: Output directory
    :param wsize: Window size for calculating validation statistics
    :param wf: Set true to write intermediate files
    :return: SWE array
    """

    swe = ssd_arr * density * 10
    if wf:
        outfile = os.path.join(outdir, 'SSD_SWE')
        np.save(outfile, swe)
        write_file(swe.copy(), img_file, outfile=outfile, is_complex=False)
    cr = check_values(swe, img_file, nsize=wsize)
    return swe, cr


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


def check_values(img_arr, img_file, geocoord=DHUNDI_COORDS, nsize=(2, 2), is_complex=False, full_stat=False):
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


def get_coherence(s1, s2, ifg, wsize, img_dict, apply_masks, verbose, wf, outdir, outfile, load_file=False):
    """
    Coherency matrix caller function
    :param s1: Master image array
    :param s2: Slave image array
    :param ifg: Interferogram array
    :param wsize: Ensemble window size (should be half of desired window size) or number of looks
    :param img_dict: Image dictionary containing GDAL references
    :param apply_masks: Set true for applying layover and forest masks
    :param verbose: Set true for detailed logs
    :param wf: Set true to save intermediate results
    :param outdir: Output directory
    :param outfile: Output file path
    :param load_file: Set true to load existing numpy binary and skip computation
    :return: Complex coherency matrix and window size string as tuple
    """

    print('Computing Coherence mat for ' + str(wsize) + '...')
    if not load_file:
        tmat = calc_ensemble_cohmat(s1, s2, ifg, apply_masks=apply_masks, outdir=outdir, outfile=outfile,
                                    img_dict=img_dict, wsize=wsize, verbose=verbose, wf=wf)
    else:
        tmat = np.load(os.path.join(outdir, 'Coherence_Ensemble_' + outfile + '.npy'))
    cr = check_values(tmat, img_dict['LIA'], geocoord=DHUNDI_COORDS, is_complex=True, nsize=wsize)
    return tmat, cr


def compute_vertical_wavenumber(lia_file, image_date, outdir, snow_density, scale_factor=1, is_single_pass=True,
                                wf=False, verbose=True, load_file=False, ensemble_avg=True, **kwargs):
    """
    Calculate vertical wavenumber, 'wsize' must be passed via kwargs to calculate validation statistics
    :param lia_file: Local incidence angle GDAL reference
    :param image_date: Image acquisition date
    :param outdir: Output directory
    :param snow_density: Snow density in gm/cm^3
    :param scale_factor: Vertical wavenumber scale factor (real valued, shoud be chosen according to the study area)
    :param is_single_pass: Set true for single-pass acquisitions
    :param wf: Set True to write intermediate files
    :param verbose: Set true for detailed logs
    :param load_file: Set true to load existing numpy binary and skip computation
    :param ensemble_avg: Set False to directly return snow depth array without ensemble averaging. If True, 
    pass appropriate wsize parameter.
    :return: Vertical wavenumber array and results over the study area cooridnates.
    """

    print('Computing vertical wavenumber ...')
    if not load_file:
        lia = get_image_array(lia_file)
        del_theta = np.abs(MEAN_INC_TDX[image_date] - MEAN_INC_TSX[image_date])
        m = 4
        if is_single_pass:
            m = 2
        eff_sqrt = np.sqrt(get_effective_permittivity(snow_density))
        kz = scale_factor * eff_sqrt * m * np.pi * np.deg2rad(del_theta) / (WAVELENGTH * np.sin(np.deg2rad(lia)))
        if wf:
            np.save(os.path.join(outdir, 'Wavenumber.npy'), kz)
        if verbose:
            print('Mean kz:', np.nanmean(kz), 'Min kz:', np.nanmin(kz), 'Max kz:', np.nanmax(kz))
        if ensemble_avg:
            kz = get_ensemble_avg(kz, outfile='KZ_Avg', wsize=kwargs['wsize'], image_file=lia_file, 
                                  verbose=verbose, wf=wf, outdir=outdir)
    else:
        if ensemble_avg:
            kz = np.load(os.path.join(outdir, 'KZ_Avg.npy'))
        else:
            kz = np.load(os.path.join(outdir, 'Wavenumber.npy'))
    cr = check_values(kz, lia_file, geocoord=DHUNDI_COORDS, is_complex=False, nsize=kwargs['wsize'])
    return kz, cr


def senstivity_analysis(image_dict, outdir, result_file, cwindows, eta_values, ct_values, scale_factors, lf_dict,
                        image_date='12292015', apply_masks=True, wf=True, verbose=False, ensemble_avg=True):
    """
    Main function for sensitivity analysis
    :param image_dict: Image dictionary containing GDAL references
    :param outdir: Output directory to store intermediate files
    :param result_file: Output file for storing analysis results
    :param cwindows: List of coherence windows for ensemble averaging
    :param eta_values: List of eta values for the hybrid DEM differencing model
    :param ct_values: List of coherence threshold values
    :param scale_factors: List of scale factors for vertical wavenumber scaling
    :param lf_dict: Dictionary containing boolean values for loading existing files
    :param image_date: Image date string (mmddyyyy) for selecting appropriate snow density
    :param apply_masks: Set true for applying layover and forest masks
    :param wf: Set False to stop writing intermediate files
    :param verbose: Set True to see all computation details
    :param ensemble_avg: Apply ensemble averaging over snow depth and vertical wavenumber
    :return: None
    """

    pol_vec = calc_pol_vec_dict()
    ifg_dir = os.path.join(outdir, 'Common')
    makedirs([ifg_dir])
    s1_vol, s2_vol, ifg_vol = calc_interferogram(image_dict, pol_vec['HV'], apply_masks=apply_masks,
                                                 outfile='Vol', verbose=verbose, load_files=lf_dict['IFG'],
                                                 outdir=ifg_dir, wf=wf)
    s1_surf, s2_surf, ifg_surf = calc_interferogram(image_dict, pol_vec['HH-VV'], apply_masks=apply_masks,
                                                    outfile='Surf', verbose=verbose, load_files=lf_dict['IFG'],
                                                    outdir=ifg_dir, wf=wf)
    lia_file = image_dict['LIA']
    for wsize in cwindows:
        w1 = wsize[0]
        w2 = wsize[1]
        wsize_half = w1 // 2, w2 // 2
        output_dir = 'CE_' + str(w1)
        if w1 != w2:
            output_dir += '_' + str(w2)
        output_dir = os.path.join(outdir, output_dir)
        makedirs([output_dir])
        tmat_vol, tvol_stats = get_coherence(s1_vol, s2_vol, ifg_vol, outfile='Vol', wsize=wsize_half,
                                             apply_masks=apply_masks, img_dict=image_dict, verbose=verbose, wf=wf, 
                                             load_file=lf_dict['COH'], outdir=output_dir)
        tmat_surf, tsurf_stats = get_coherence(s1_surf, s2_surf, ifg_surf, outfile='Surf', wsize=wsize_half,
                                               apply_masks=apply_masks, img_dict=image_dict, verbose=verbose, wf=wf,
                                               load_file=lf_dict['COH'], outdir=output_dir)
        ground_phase = get_ground_phase(tmat_vol, tmat_surf, wsize=wsize_half, img_dict=image_dict,
                                        apply_masks=apply_masks, verbose=verbose, wf=wf, load_file=lf_dict['GP'],
                                        outdir=output_dir)
        snow_density = STANDING_SNOW_DENSITY[image_date]
        kz, kz_stats = compute_vertical_wavenumber(lia_file, snow_density=snow_density, image_date=image_date,
                                                   verbose=verbose, load_file=lf_dict['KZ'], outdir=output_dir, wf=wf,
                                                   ensemble_avg=ensemble_avg, wsize=wsize_half)
        for eta in eta_values:
            for ct in ct_values:
                snow_depth, sd_stats = calc_snow_depth_hybrid(tmat_vol, ground_phase, kz, img_file=lia_file, eta=eta,
                                                              coherence_threshold=ct, wf=wf, load_file=lf_dict['SD'],
                                                              outdir=output_dir, ensemble_avg=ensemble_avg,
                                                              wsize=wsize_half, verbose=verbose)
                for sf in scale_factors:
                    snow_depth_scaled = snow_depth / sf
                    swe, swe_stats = get_total_swe(snow_depth_scaled, density=snow_density, img_file=lia_file,
                                                   outdir=output_dir, wf=wf, wsize=wsize_half)
                    ssd_actual = STANDING_SNOW_DEPTH[image_date]
                    sswe_actual = ssd_actual * snow_density * 10
                    img_date = pd.to_datetime(image_date, format='%m%d%Y')
                    result_dict = {'Date': [img_date], 'CWindow': [str(wsize)], 'Eta': [eta], 'CT': [ct], 'SF': [sf],
                                   'Mean_SSD_Est(cm)': [sd_stats[0] / sf], 'SD_SSD_Est(cm)': [sd_stats[1] / sf],
                                   'Mean_SSWE_Est(mm)': [swe_stats[0]], 'SD_SSWE_Est(mm)': [swe_stats[1]],
                                   'SSD_Actual(cm)': [ssd_actual], 'SSWE_Actual(mm)': [sswe_actual],
                                   'Mean_TVOL': [tvol_stats[0]], 'SD_TVOL': [tvol_stats[1]],
                                   'Mean_TSURF': [tsurf_stats[0]], 'SD_TSURF': [tsurf_stats[1]],
                                   'Mean_KZ(rad/cm)': [kz_stats[0] * sf], 'SD_KZ(rad/cm)': [kz_stats[1] * sf],
                                   'Pass': [ACQUISITION_ORIENTATION[image_date]]}
                    print(result_dict)
                    df = pd.DataFrame(data=result_dict)
                    df.to_csv(result_file, sep=';', index=False, mode='a')


def coherence_analysis():
    """
    Compare surface coherence between summer and winter periods
    :return: None
    """

    coh1 = gdal.Open('PolinSnow_Data/Coherence/Coh_01082016.tif')
    coh2 = gdal.Open('PolinSnow_Data/Coherence/Coh_06082017.tif')
    layover_file = gdal.Open(r'PolinSnow_Data\Common\DESC\LAYOVER.tif')
    forest_file = gdal.Open(r'PolinSnow_Data\Common\FOREST.tif')

    coh1_arr = get_image_array(coh1, no_data_value=-32767)
    coh2_arr = get_image_array(coh2, no_data_value=-32767)
    coh1_arr = set_nan_img(coh1_arr, layover_file=layover_file, forest_file=forest_file)
    coh2_arr = set_nan_img(coh2_arr, layover_file=layover_file, forest_file=forest_file)
    print(check_values(coh1_arr, coh1))
    print(check_values(coh2_arr, coh2))
    print(np.nanmean(coh1_arr))
    print(np.nanmean(coh2_arr))


def makedirs(directory_list):
    """
    Create directory for storing files
    :param directory_list: List of directories to create
    :return: None
    """

    for directory_name in directory_list:
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)


def run_polinsnow(lf_ifg=True, lf_coh=True, lf_gp=True, lf_kz=False, lf_sd=False, wf=True):
    """
    Initializes all hyperparameters and performs sensitivity analyses
    :param lf_ifg: Set True to Load existing files related to the PolInSAR interferogram
    :param lf_coh: Set True to Load existing files related to the PolInSAR coherence
    :param lf_gp: Set True to Load existing files related to the PolInSAR ground phase
    :param lf_kz: Set True to Load existing files related to the PolInSAR vertical wavenumber
    :param lf_sd: Set True to Load existing files related to the PolInSAR snow depth
    :param wf: Set False to disable file writing
    :return: None
    """

    image_dates = list(ACQUISITION_ORIENTATION.keys())
    completed = []
    lf_dict = {'IFG': lf_ifg, 'COH': lf_coh, 'GP': lf_gp, 'KZ': lf_kz, 'SD': lf_sd}
    result_dir = 'Analysis_Results'
    makedirs([result_dir])
    result_file = os.path.join(result_dir, 'Sensitivity_Results_GIS_New.csv')
    if os.path.exists(result_file):
        os.remove(result_file)
    n_jobs = min(len(image_dates), multiprocessing.cpu_count())
    if lf_ifg and lf_coh and lf_gp and lf_kz and lf_sd:
        n_jobs = 1
    Parallel(n_jobs=n_jobs)(delayed(parallel_compute)(image_date=image_date, completed=completed, lf_dict=lf_dict,
                                                      result_file=result_file, wf=wf) for image_date in image_dates)
    df = pd.read_csv(result_file, sep=';')
    df.sort_values('Date')
    df = df.drop_duplicates(keep=False)
    df = df.dropna()
    updated_file = os.path.join(result_dir, 'Sensitivity_Results_GIS_T1_New.csv')
    df.to_csv(updated_file, sep=';', index=False)


def parallel_compute(image_date, completed, lf_dict, result_file, wf):
    """
    Use parallel computation, must be called from inside #run_polinsnow()
    :param image_date: Image acquisition data
    :param completed: List of completed image dates
    :param lf_dict: Dictionary containing boolean values for loading existing files
    :param result_file: Output result file
    :param wf: Set False to disable file writing
    :return: None
    """

    warnings.filterwarnings("ignore")
    if image_date not in completed:
        print('Working with', image_date, 'data...\n')
        base_path = 'PolinSnow_Data'
        image_path = os.path.join(base_path, image_date)
        common_path = os.path.join(base_path, 'Common')
        output_path = os.path.join('Outputs_New_T2', image_date)
        image_dict = read_images(image_path=image_path, common_path=common_path, verbose=False)
        w = list(range(5, 56, 10)) + [21]
        windows = list(zip(w, w))
        eta_values = [0.6]
        ct_values = [0.]
        scale_factors = range(1, 101)
        senstivity_analysis(image_dict, cwindows=windows, eta_values=eta_values, ct_values=ct_values,
                            scale_factors=scale_factors, image_date=image_date, outdir=output_path, lf_dict=lf_dict,
                            ensemble_avg=True, verbose=False, result_file=result_file, wf=wf)


run_polinsnow(lf_ifg=True, lf_coh=False, lf_gp=False, lf_kz=False, lf_sd=False, wf=True)
