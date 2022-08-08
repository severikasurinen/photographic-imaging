# TODO: Comments
import settings

import math
import colour
import scipy.interpolate as scint
import os
import time
import multiprocessing
from itertools import repeat
import numpy as np
import csv
from matplotlib import pyplot as plt
plt.rcParams["figure.dpi"] = 200


def create_directories(main_dir):
    folders = (r'..\Images', r'Calibration\Image Uniformity', r'Calibration\Correction Profiles',
               r'Calibration\ICC Profiles', r'Calibration\Calibration Images', 'Corrected Images', 'Exported Images',
               'Remote Capture')
    for folder in folders:
        path = os.path.join(main_dir, folder)
        if not os.path.exists(path):
            os.makedirs(path)


# Colored print to console
def print_color(in_str, in_col):
    print(settings.print_colors[in_col] + in_str + settings.print_colors['end'])


# Write data to .csv file
def write_csv(in_data, in_name, sub_path):
    csv_path = os.path.join(settings.main_directory, sub_path)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    with open(os.path.join(csv_path, in_name + '.csv'), 'w', encoding='UTF8', newline='') as f:
        for i in range(len(in_data)):
            for o in range(len(in_data[i])):
                if type(in_data[i][o]) is list:
                    in_data[i][o] = ' '.join(str(val) for val in in_data[i][o])

        writer = csv.writer(f)
        writer.writerows(in_data)


# Read data from .csv file
def read_csv(in_name, sub_path):
    csv_path = os.path.join(settings.main_directory, sub_path)
    if not os.path.exists(os.path.join(csv_path, in_name + '.csv')):
        return None

    headers = None
    out_data = []
    with open(os.path.join(csv_path, in_name + '.csv'), 'r', encoding='UTF8', newline='') as f:

        for row in csv.reader(f, delimiter=','):
            raw_cells = []
            for cell in row:
                raw_cells.append(cell.split(' '))
            cells = []
            for i in range(len(raw_cells)):
                cells.append([])
                for o in range(len(raw_cells[i])):
                    raw_cells[i][o] = raw_cells[i][o].strip()
                    if raw_cells[i][o] != '':
                        try:
                            raw_cells[i][o] = float(raw_cells[i][o])
                        except:
                            pass
                        cells[i].append(raw_cells[i][o])

            if len(cells) == 1:
                cells = cells[0]
            if headers is None:
                headers = cells
            else:
                out_data.append(cells)

    return headers, out_data


# Compute 3D LUT from reference data
def create_lut(in_lab, in_res, in_domain):
    edges = []
    for i in range(3):
        edges.append(np.linspace(in_domain[0][i], in_domain[1][i], settings.lut_size))
    lab_i = np.meshgrid(edges[0], edges[1], edges[2])
    # split_num = math.ceil(len(edges[0]) / math.pow(settings.calib_max_vals, 1 / 3))
    # while len(edges[0]) % split_num != 0:
    #    split_num += 1
    # split_len = int(len(edges[0]) / split_num)

    rbf_func = []
    # res_i = []
    for i in range(3):
        rbf_func.append(scint.Rbf(in_lab[0], in_lab[1], in_lab[2], in_res[i], epsilon=settings.correction_epsilon))

    # print(np.array(lab_i).shape)
    # for sub_grid in np.array(lab_i).reshape(-1, 3, split_len, split_len, split_len):
    #    #print(sub_grid.shape)
    #    res_i.append((rbf_func[0](sub_grid[0], sub_grid[1], sub_grid[2]),
    #                  rbf_func[1](sub_grid[0], sub_grid[1], sub_grid[2]),
    #                  rbf_func[2](sub_grid[0], sub_grid[1], sub_grid[2])))
    # print(np.array(res_i).shape)

    # print(rbf_func[0](20, 0, 0), rbf_func[0](50, 0, 0), rbf_func[0](80, 0, 0))

    # interp_table = np.concatenate(res_i, axis=0).reshape(3, len(edges[0]), len(edges[0]), len(edges[0])).T
    # print(interp_table.shape)
    interp_table = (np.clip(rbf_func[0](lab_i[0], lab_i[1], lab_i[2]), 0, 100),
                    np.clip(rbf_func[1](lab_i[0], lab_i[1], lab_i[2]), -100, 100),
                    np.clip(rbf_func[2](lab_i[0], lab_i[1], lab_i[2]), -100, 100))
    interp_table = np.concatenate(np.array(interp_table).T, axis=-1).reshape((len(edges[0]),
                                                                              len(edges[0]),
                                                                              len(edges[0]), 3))

    out_lut = colour.LUT3D(interp_table, 'correction', in_domain, settings.lut_size)
    # print(out_lut.apply(((20, 0, 0), (50, 0, 0), (80, 0, 0))))

    return out_lut


# Plot multiple spectral distribution graphs
def plot_spectra(spectrum_files, sub_path, spectrum_domain=(400, 800), intensity_range=(0, 8000), base_spectrum=None,
                 sum_indices=()):
    csv_headers = ('Wavelength λ (nm)', 'Intensity (counts)')   # Default headers
    axis_ranges = (spectrum_domain[0], spectrum_domain[1], intensity_range[0], intensity_range[1])
    sds = []
    base_noise = {}
    if base_spectrum is not None:
        measurement_count = {}
        csv_headers, spectrum_csv = read_csv(base_spectrum, sub_path)
        for i in range(len(spectrum_csv)):
            elem = (round(float(str(spectrum_csv[i][0]).split('\t')[0])), float(str(spectrum_csv[i][0]).split('\t')[1]))
            if elem[0] in base_noise:
                base_noise[elem[0]] += elem[1]
            else:
                base_noise[elem[0]] = elem[1]
            if elem[0] in measurement_count:
                measurement_count[elem[0]] += 1
            else:
                measurement_count[elem[0]] = 1

        for key in base_noise:
            base_noise[key] /= measurement_count[key]

    for i in range(len(spectrum_files)):
        spectrum_name = spectrum_files[i].split('_')[1] + ' min'
        if len(spectrum_files[i].split('_')) > 2:
            spectrum_name += ', ' + spectrum_files[i].split('_')[2]

        measurement_count = {}
        csv_headers, spectrum_csv = read_csv(spectrum_files[i], sub_path)
        spectrum_data = {}
        avg_noise = [0, 0]
        for o in range(len(spectrum_csv)):
            elem = (float(str(spectrum_csv[o][0]).split('\t')[0]), float(str(spectrum_csv[o][0]).split('\t')[1]))

            if elem[0] < spectrum_domain[0] or elem[0] > spectrum_domain[1]:
                avg_noise[0] += 1
                avg_noise[1] += elem[1]
            if round(elem[0]) in measurement_count:
                measurement_count[round(elem[0])] += 1
            else:
                measurement_count[round(elem[0])] = 1

        avg_noise = avg_noise[1] / avg_noise[0]

        for o in range(len(spectrum_csv)):
            elem = (round(float(str(spectrum_csv[o][0]).split('\t')[0])), float(str(spectrum_csv[o][0]).split('\t')[1]))
            if base_spectrum is None:
                cur_noise = avg_noise
            else:
                cur_noise = base_noise[elem[0]]
            if spectrum_domain[0] - 1 <= elem[0] <= spectrum_domain[1] + 1:
                if elem[0] not in spectrum_data:
                    spectrum_data[elem[0]] = (elem[1] - cur_noise) / measurement_count[elem[0]]
                else:
                    spectrum_data[elem[0]] = spectrum_data[elem[0]] + (elem[1] - cur_noise) / measurement_count[elem[0]]

        sds.append(colour.SpectralDistribution(spectrum_data, name=f"{spectrum_name}"))
        colour.plotting.plot_single_sd(sds[i], title=f"{'Spectral Distribution'} ({spectrum_name})",
                                       x_label=csv_headers[0], y_label=csv_headers[1], bounding_box=axis_ranges)

    if len(sum_indices) > 1:
        spectrum_data = {}
        for key in sds[0].wavelengths:
            spectrum_data[key] = 0
            for i in sum_indices:
                spectrum_data[key] += sds[i][key]

        sds.append(colour.SpectralDistribution(spectrum_data, name=f"sum of {sum_indices}"))
        colour.plotting.plot_single_sd(sds[-1], title=f"{'Spectral Distribution'} (sum of {sum_indices})",
                                       x_label=csv_headers[0], y_label=csv_headers[1], bounding_box=axis_ranges)

    colour.plotting.plot_multi_sds(sds, title='Spectral Distribution', x_label=csv_headers[0], y_label=csv_headers[1],
                                   bounding_box=axis_ranges)


def get_angle(start_point, end_point, radians=False, clamp=False):  # Get angle between 2 points
    # Get angle in range [-180, 180]
    angle = math.degrees(math.atan2((end_point[1] - start_point[1]), end_point[0] - start_point[0]))

    if clamp:  # Get angle in range [0, 360]
        if angle < 0:
            angle += 360

    if radians:
        return math.degrees(angle)
    else:
        return angle


# Get dict key by value
def get_key(in_val, in_dict):
    for key in in_dict.keys():
        if in_val == in_dict[key]:
            return key
    return None


def parallel_operation(in_array, const, operation=0, is_thread=False):  # operation - 0: sum, 1: multiply, 2: power
    if is_thread or settings.cpu_threads == 0:
        if operation == 0:
            return in_array + const
        elif operation == 1:
            return np.multiply(in_array, const)
        elif operation == 2:
            return np.power(in_array, const)
    else:
        split_array = np.array_split(in_array, min(settings.cpu_threads, 16))

        pool = multiprocessing.pool.ThreadPool()
        out_array = pool.starmap(parallel_operation, zip(split_array, repeat(const), repeat(operation), repeat(True)))

        return np.concatenate(out_array)


# Print rough operation estimate
def print_estimate(start_time, progress):
    est_time = round((time.perf_counter() - start_time) * (1 / progress - 1))
    if est_time > 5:
        print()
        if est_time < 60:
            print("Estimated time left:", str(est_time), "s")
        else:
            print("Estimated time left:", str(round(est_time / 60)), "min", str(round(est_time % 60)), "s")
        print()


# Wait for Y/N answer
def yes_no_prompt(prompt):
    while True:
        cur_input = input(f"{prompt} (Y/N): ")
        if cur_input is not None:
            cur_input = cur_input.strip().capitalize()
            if cur_input == 'Y':
                return True
            elif cur_input == 'N':
                return False

        print("Invalid input, only use Y/N.")
