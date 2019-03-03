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
        arr[arr == no_data_value] = np.nan
    return arr


def get_classified_aspect(aspect_val):
    """
    Get slope direction
    :param aspect_val: Aspect value in degrees
    :return: Slope direction
    """

    if aspect_val <= -1:
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

    if slope_val < 0:
        return 'NaN'
    elif slope_val <= 20:
        return 'S1'
    elif 20 < slope_val <= 40:
        return 'S2'
    return 'S3'


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


def get_dict_stat(count_dict, val_dict=None, res=3, only_area=False):
    """
    Calculate mean, standard deviation and standard error of snow depth or
    snow covered area (SCA) in sq. km from dictionary
    :param only_area: Set true to calculate only area statistics
    :param val_dict: Dictionary containing values
    :param count_dict: Dictionary containing number of snow pixels per class
    :param res: Pixel resolution in metres
    :return: Area dict and/or Stat dict
    """

    stat_dict = {}
    area_dict = {}
    for key in count_dict.keys():
        if not only_area:
            sd_arr = np.array(val_dict[key])
            mean_sd = np.mean(sd_arr)
            std_dev = np.std(sd_arr)
            std_err = std_dev / np.sqrt(sd_arr.size)
            stat_dict[key] = (np.round(mean_sd, 2), np.round(std_dev, 2), np.round(std_err, 2))
        area_dict[key] = np.round((count_dict[key] * res ** 2) / 1E+6, 2)
    if only_area:
        return area_dict
    return area_dict, stat_dict


def calc_scattering_dict(wishart_arr, img_dict, scat_values=tuple(range(1, 10))):
    """
    Calculate wishart class statistics wrt aspect, elevation and slope
    :param scat_values: Wishart scattering classes
    :param wishart_arr: Wishart classified array
    :param img_dict: Image dictionary containing GDAL references
    :return: None
    """

    aspect_arr = get_image_array(img_dict['ASPECT'], set_no_data=False)
    elevation_arr = get_image_array(img_dict['ELEVATION'], set_no_data=False)
    slope_arr = get_image_array(img_dict['SLOPE'], set_no_data=False)

    for wc in scat_values:
        count_aspect = defaultdict(lambda: 0)
        count_elevation = defaultdict(lambda: 0)
        count_slope = defaultdict(lambda: 0)
        for idx, val in np.ndenumerate(wishart_arr):
            if val == wc:
                aspect_class = get_classified_aspect(aspect_arr[idx])
                elevation_class = get_classified_elevation(elevation_arr[idx])
                slope_class = get_classified_slope(slope_arr[idx])
                count_aspect[aspect_class] += 1
                count_elevation[elevation_class] += 1
                count_slope[slope_class] += 1

        print('\nScattering area for class wrt aspect', wc)
        area_dict = get_dict_stat(count_aspect, only_area=True)
        print(area_dict)
        print('\nScattering area for class wrt elevation', wc)
        area_dict = get_dict_stat(count_elevation, only_area=True)
        print(area_dict)
        print('\nScattering area for class wrt slope', wc)
        area_dict = get_dict_stat(count_slope, only_area=True)
        print(area_dict)


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

    la_aspect = get_dict_stat(count_layover_aspect, only_area=True)
    la_elevation = get_dict_stat(count_layover_elevation, only_area=True)
    la_slope = get_dict_stat(count_layover_slope, only_area=True)
    print('Layover area (sq. km)')
    print('LA:', la_aspect)
    print('LE:', la_elevation)
    print('LS:', la_slope)

    fa_aspect = get_dict_stat(count_forest_aspect, only_area=True)
    fa_elevation = get_dict_stat(count_forest_elevation, only_area=True)
    fa_slope = get_dict_stat(count_forest_slope, only_area=True)

    print('Forest area (sq. km)')
    print('\nFA', fa_aspect)
    print('FE', fa_elevation)
    print('FS', fa_slope)


def calc_areas(img_file, type='A'):
    """
    Calculate Aspect, Elevation or Slope areas
    :param img_file: GDAL reference corresponding to aspect, elevation or slope
    :param type: Set 'A' for aspect, 'E' for elevation, and 'S' for slope
    :return: None
    """

    set_no_data = True
    if type == 'A' or type == 'S':
        set_no_data = False
    img_arr = get_image_array(img_file, set_no_data=set_no_data)
    area_dict = defaultdict(lambda: 0)
    count = 0
    for index, val in np.ndenumerate(img_arr):
        if val != -32767 or not np.isnan(val):
            count += 1
            if type == 'A':
                c_type = get_classified_aspect(val)
            elif type == 'E':
                c_type = get_classified_elevation(val)
            else:
                c_type = get_classified_slope(val)
            area_dict[c_type] += 1
    area_stat = get_dict_stat(area_dict, only_area=True)
    print(count * 9 / 1E+6)
    print(type + ' Area in sq. km:')
    print(area_stat)


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

    aspect_dict = defaultdict(lambda: [])
    elevation_dict = defaultdict(lambda: [])
    slope_dict = defaultdict(lambda: [])
    count_aspect = defaultdict(lambda: 0)
    count_elevation = defaultdict(lambda: 0)
    count_slope = defaultdict(lambda: 0)

    for idx, sd in np.ndenumerate(sd_arr):
        if not np.isnan(sd):
            aspect_class = get_classified_aspect(aspect_arr[idx])
            elevation_class = get_classified_elevation(elevation_arr[idx])
            slope_class = get_classified_slope(slope_arr[idx])
            aspect_dict[aspect_class].append(sd)
            elevation_dict[elevation_class].append(sd)
            slope_dict[slope_class].append(sd)
            count_aspect[aspect_class] += 1
            count_elevation[elevation_class] += 1
            count_slope[slope_class] += 1

    sca_aspect, aspect_dict = get_dict_stat(count_aspect, aspect_dict)
    sca_elevation, elevation_dict = get_dict_stat(count_elevation, elevation_dict)
    sca_slope, slope_dict = get_dict_stat(count_slope, slope_dict)

    print('\nSD_Values (cm)')
    print('A:', aspect_dict)
    print('E:', elevation_dict)
    print('S:', slope_dict)

    print('\nSCA (sq. km)')
    print('A:', sca_aspect)
    print('E:', sca_elevation)
    print('S:', sca_slope)


def write_file(arr, src_file, outfile='test', is_complex=False, no_data_value=-32768, dt=gdal.GDT_Float32):
    """
    Write image files in TIF format
    :param dt: Datatype of output file
    :param arr: Image array to write
    :param src_file: Original image file for retrieving affine transformation parameters
    :param outfile: Output file path
    :param no_data_value: No data value to be set
    :param is_complex: If true, write complex image array in two separate bands
    :return: None
    """

    driver = gdal.GetDriverByName("GTiff")
    if is_complex:
        out = driver.Create(outfile + ".tif", arr.shape[1], arr.shape[0], 2, dt)
    else:
        out = driver.Create(outfile + ".tif", arr.shape[1], arr.shape[0], 1, dt)
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


def get_wishart_class_stats(wishart_arr, layover_arr, forest_arr, outfile, img_file, check_forests,
                            total_pixels=None):
    """
    Calculate Wishart class percentages
    :param total_pixels: This is useful when two images are misaligned by a few pixels
    :param img_file: Original GDAL reference containing affine transformation coordinates
    :param outfile: Output file name
    :param check_forests: Set true to mask out forests
    :param forest_arr: Forest array
    :param wishart_arr: Wishart classified image array
    :param layover_arr: Layover array
    :return: None
    """

    new_arr = wishart_arr.copy()
    new_arr.fill(np.nan)
    print('Checking valid pixels...')
    for index, value in np.ndenumerate(wishart_arr):
        if not np.isnan(value):
            new_arr[index] = int(round(value))
            if new_arr[index] == 0:
                new_arr[index] = 1
            if np.round(layover_arr[index]) != 0 or (check_forests and forest_arr[index] == 0):
                new_arr[index] = np.nan
    write_file(new_arr.copy(), img_file, outfile=outfile)
    new_arr = new_arr[~np.isnan(new_arr)]
    classes, count = np.unique(new_arr, return_counts=True)
    if not total_pixels:
        total_pixels = np.sum(count)
    print('Total pixels=', total_pixels)
    class_percent = np.round(count * 100. / total_pixels, 2)
    print(classes, class_percent)
    return total_pixels


def correct_wishart_file(img_dict, check_forests=False):
    """
    Convert fuzzy wishart classes to crisp
    :param check_forests: Set true to apply forest mask
    :param img_dict: Image dictionary containing GDAL references
    :return: None
    """
    w1_file = img_dict['Wishart_Jan_Quad']
    w1_arr = get_image_array(w1_file)
    # w2_arr = get_image_array(img_dict['Wishart_Jun'])

    layover_arr = get_image_array(img_dict['LAYOVER'])
    forest_arr = None
    if check_forests:
        forest_arr = get_image_array(img_dict['FOREST'])
    print('\nWishart_Jan')
    total_pixels = get_wishart_class_stats(w1_arr, layover_arr, forest_arr, outfile='Out/WJan_Quad', img_file=w1_file,
                                           check_forests=check_forests)
    # print('\nWishart_June')
    # get_wishart_class_stats(w2_arr, layover_arr, forest_arr, outfile='Out/WJun', img_file=w1_file,
    #                        check_forests=check_forests, total_pixels=total_pixels)


img_dict = read_images('/home/iirs/THESIS/Thesis_Files/Snow_Analysis/', '*.tif')
# calc_mask_dict(img_dict)
fsd_arr = get_image_array(img_dict['FSD'])
ssd_arr = get_image_array(img_dict['SSD'])
print('FSD stats')
calc_sd_dict(img_dict, fsd_arr)
print('\nSSD stats')
calc_sd_dict(img_dict, ssd_arr)
# wjan_arr = get_image_array(img_dict['WJan'])
# wjun_arr = get_image_array(img_dict['WJun'])
# print('\nWishart Jan stats')
# calc_scattering_dict(wjan_arr, img_dict, scat_values=(2, 3, 5, 8))
# print('\nWishart Jun stats')
# calc_scattering_dict(wjun_arr, img_dict, scat_values=(3, 5, 8))
# correct_wishart_file(img_dict)
# calc_areas(img_dict['ASPECT'], type='A')