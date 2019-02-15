import gdal
import numpy as np
import glob
import os
from collections import defaultdict


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


def get_image_array(img_file, set_no_data=True):
    """
    Read real numpy arrays from file
    :param set_no_data: Set False to not set nan values
    :param img_file: GDAL reference file
    :return: Numpy array with nan set accordingly
    """

    band = img_file.GetRasterBand(1)
    no_data_value = band.GetNoDataValue()
    arr = band.ReadAsArray()
    if set_no_data:
        arr[arr == int(no_data_value)] = np.nan
    return arr


def get_classified_aspect(aspect_val):
    """
    Get slope direction
    :param aspect_val: Aspect value in degrees
    :return: Slope direction
    """

    if aspect_val == -1:
        return 'F'
    elif aspect_val <= 22.5 or aspect_val >= 337.5:
        return 'N'
    elif 22.5 < aspect_val <= 67.5:
        return 'NE'
    elif 67.5 < aspect_val <= 112.5:
        return 'E'
    elif 112.5 < aspect_val <= 157.5:
        return 'SE'
    elif 157.5 < aspect_val <= 202.5:
        return 'S'
    elif 202.5 < aspect_val <= 247.5:
        return 'SW'
    elif 247.5 < aspect_val <= 292.5:
        return 'W'
    return 'NW'


def get_classified_slope(slope_val):
    """
    Classify slope value
    :param slope_val: Slope value in degrees
    :return: Classified slope
    """

    if slope_val <= 20:
        return 'L'
    elif 20 < slope_val < 40:
        return 'M'
    return 'H'


def get_classified_elevation(elevation_val):
    """
    Classify elevation value
    :param elevation_val: Elevation value in metres
    :return: Classified elevation
    """

    if elevation_val <= 2500:
        return 'E1'
    elif 2500 < elevation_val <= 3000:
        return 'E2'
    elif 3000 < elevation_val <= 3500:
        return 'E3'
    elif 3500 < elevation_val <= 4000:
        return 'E4'
    elif 4000 < elevation_val <= 4500:
        return 'E5'
    return 'E6'


def get_mask_stat(mask_dict, res=3):
    """
    Get mask areas in sq. km
    :param mask_dict: Dictionary containing mask values
    :param res: Pixel resolution in metres
    :return: Mask area dict
    """

    area_dict = {}
    for key in mask_dict.keys():
        area_dict[key] = np.round(mask_dict[key] * res ** 2 / 1E+6, 2)
    return area_dict


def get_sd_dict_stat(sd_dict, count_dict, res=3):
    """
    Calculate mean snow depths and snow covered area (SCA) in sq. km from dictionary
    :param sd_dict: Dictionary containing total snow depth values
    :param count_dict: Dictionary containing number of snow pixels per class
    :param res: Pixel resolution in metres
    :return: Mean SD dict and SCA dict
    """

    mean_dict = {}
    sca_dict = {}
    for key in sd_dict.keys():
        mean_dict[key] = np.round(sd_dict[key] / count_dict[key], 2)
        sca_dict[key] = np.round((count_dict[key] * res ** 2) / 1E+6, 2)
    return mean_dict, sca_dict


def calc_mask_dict(img_dict):
    """
    Calculate mask dictionary stats
    :param img_dict: Image dictionary containing GDAL references
    :return: None
    """

    layover_arr = get_image_array(img_dict['LAYOVER'])
    forest_arr = get_image_array(img_dict['FOREST'])
    aspect_arr = get_image_array(img_dict['ASPECT'], set_no_data=False)
    elevation_arr = get_image_array(img_dict['ELEVATION'], set_no_data=False)
    slope_arr = get_image_array(img_dict['SLOPE'], set_no_data=False)

    count_layover_aspect = defaultdict(lambda: 0)
    count_layover_elevation = defaultdict(lambda: 0)
    count_layover_slope = defaultdict(lambda: 0)
    count_forest_aspect = defaultdict(lambda: 0)
    count_forest_elevation = defaultdict(lambda: 0)
    count_forest_slope = defaultdict(lambda: 0)

    larr = layover_arr[~np.isnan(layover_arr)]
    tarea = len(larr) * 9 / 1E+6
    larea = len(larr[np.round(larr) != 0]) * 9 / 1E+6
    farea = len(forest_arr[forest_arr == 0]) * 9 / 1E+6
    print('Total study area', tarea)
    print('Total layover area', larea, '%=', larea * 100 / tarea)
    print('Total forest area', farea, '%=', farea * 100 / tarea)

    for idx, lval in np.ndenumerate(layover_arr):
        if not np.isnan(lval):
            aspect_class = get_classified_aspect(aspect_arr[idx])
            elevation_class = get_classified_elevation(elevation_arr[idx])
            slope_class = get_classified_slope(slope_arr[idx])
            if np.round(layover_arr[idx]) != 0:
                count_layover_aspect[aspect_class] += 1
                count_layover_elevation[elevation_class] += 1
                count_layover_slope[slope_class] += 1
            if forest_arr[idx] == 0:
                count_forest_aspect[aspect_class] += 1
                count_forest_elevation[elevation_class] += 1
                count_forest_slope[slope_class] += 1

    la_aspect = get_mask_stat(count_layover_aspect)
    la_elevation = get_mask_stat(count_layover_elevation)
    la_slope = get_mask_stat(count_layover_slope)

    print('Layover area (sq. km)')
    print('LA:', la_aspect)
    print('LE:', la_elevation)
    print('LS:', la_slope)

    fa_aspect = get_mask_stat(count_forest_aspect)
    fa_elevation = get_mask_stat(count_forest_elevation)
    fa_slope = get_mask_stat(count_forest_slope)

    print('Forest area (sq. km)')
    print('\nFA', fa_aspect)
    print('FE', fa_elevation)
    print('FS', fa_slope)


def calc_sd_dict(img_dict, sd_arr):
    """
    Calculate snow depth variation towards elevation, slope, and aspect
    :param sd_arr: Snow depth array
    :param img_dict: Dictionary containing GDAL references
    :return: None
    """

    aspect_arr = get_image_array(img_dict['ASPECT'], set_no_data=False)
    elevation_arr = get_image_array(img_dict['ELEVATION'], set_no_data=False)
    slope_arr = get_image_array(img_dict['SLOPE'], set_no_data=False)

    aspect_dict = defaultdict(lambda: 0)
    elevation_dict = defaultdict(lambda: 0)
    slope_dict = defaultdict(lambda: 0)
    count_aspect = defaultdict(lambda: 0)
    count_elevation = defaultdict(lambda: 0)
    count_slope = defaultdict(lambda: 0)

    for idx, sd in np.ndenumerate(sd_arr):
        if not np.isnan(sd):
            aspect_class = get_classified_aspect(aspect_arr[idx])
            elevation_class = get_classified_elevation(elevation_arr[idx])
            slope_class = get_classified_slope(slope_arr[idx])
            aspect_dict[aspect_class] += sd
            elevation_dict[elevation_class] += sd
            slope_dict[slope_class] += sd
            count_aspect[aspect_class] += 1
            count_elevation[elevation_class] += 1
            count_slope[slope_class] += 1

    aspect_dict, sca_aspect = get_sd_dict_stat(aspect_dict, count_aspect)
    elevation_dict, sca_elevation = get_sd_dict_stat(elevation_dict, count_elevation)
    slope_dict, sca_slope = get_sd_dict_stat(slope_dict, count_slope)

    print('\nSD_Values (cm)')
    print('A:', aspect_dict)
    print('E:', elevation_dict)
    print('S:', slope_dict)

    print('\nSCA (sq. km)')
    print('A:', sca_aspect)
    print('E:', sca_elevation)
    print('S:', sca_slope)


img_dict = read_images('/home/iirs/THESIS/Thesis_Files/Snow_Analysis/', '*.tif')
calc_mask_dict(img_dict)
# fsd_arr = get_image_array(img_dict['FSD'])
# ssd_arr = get_image_array(img_dict['SSD'])
# print('FSD stats')
# calc_sd_dict(img_dict, fsd_arr)
# print('\nSSD stats')
# calc_sd_dict(img_dict, ssd_arr)






