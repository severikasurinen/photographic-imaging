import settings

import os
import time
import multiprocessing
import math
import csv
import colour
import scipy.interpolate as scint
import natsort
from itertools import repeat
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.dpi"] = 200


# Create program directory structure
def create_directories(main_dir):
    for folder in settings.directories:
        path = os.path.join(main_dir, folder)
        if not os.path.exists(path):
            os.makedirs(path)


# List files in given directory
def get_files(sub_path, match_extension=None):
    file_names = natsort.natsorted(os.listdir(os.path.join(settings.main_directory, sub_path)))  # Get files in order

    if match_extension is None:
        # Ignore folders
        return [file_name for file_name in file_names if len(str(file_name).split('.')) > 1]
    else:
        # Ignore folders and check for given extension
        return [file_name for file_name in file_names
                if len(str(file_name).split('.')) > 1 and str(file_name).split('.')[1] == match_extension]


def print_color(in_str, in_col):
    """Colored print to console"""
    if settings.use_colored_printing:
        print(settings.print_colors[in_col] + in_str + settings.print_colors['end'])
    else:
        print(in_str)


def write_csv(in_data, in_name, sub_path):
    """Write data to .csv file"""
    csv_path = os.path.join(settings.main_directory, sub_path)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    with open(os.path.join(csv_path, in_name + '.csv'), 'w', encoding='UTF8', newline='') as f:
        for i in range(len(in_data)):
            for o in range(len(in_data[i])):
                if type(in_data[i][o]) is list:
                    in_data[i][o] = ' '.join(str(val) for val in in_data[i][o])     # 3D separated by spaces

        writer = csv.writer(f)
        writer.writerows(in_data)


def read_csv(in_name, sub_path):
    """Read data from .csv file"""
    csv_path = os.path.join(settings.main_directory, sub_path)
    if not os.path.exists(os.path.join(csv_path, in_name + '.csv')):
        return None

    headers = None
    out_data = []
    with open(os.path.join(csv_path, in_name + '.csv'), 'r', encoding='UTF8', newline='') as f:

        for row in csv.reader(f, delimiter=','):
            if headers is None:
                headers = row   # Read headers from first row
                continue
            raw_cells = []
            for cell in row:
                raw_cells.append(cell.split(' '))   # Split 3D data
            cells = []
            for i in range(len(raw_cells)):
                cells.append([])
                for o in range(len(raw_cells[i])):
                    val = raw_cells[i][o].strip()   # Remove spaces
                    if val != '':                   # Ignore if empty
                        try:
                            val = float(val)        # Convert to float if possible
                        except ValueError:
                            pass
                        cells[i].append(val)
                if len(cells[i]) == 1:
                    cells[i] = cells[i][0]

            if len(cells) == 1:
                cells = cells[0]
            out_data.append(cells)

    return headers, out_data


def create_lut(in_lab, in_res, in_domain):
    """Compute 3D LUT from reference data"""
    edges = []
    # Create grid of points for LUT
    for i in range(3):
        edges.append(np.linspace(in_domain[0][i], in_domain[1][i], settings.lut_size))
    lab_i = np.meshgrid(edges[0], edges[1], edges[2])

    # Create RBF functions for color interpolation & extrapolation
    rbf_func = []
    for i in range(3):
        rbf_func.append(scint.Rbf(in_lab[0], in_lab[1], in_lab[2], in_res[i], epsilon=settings.correction_epsilon))

    interp_table = (np.clip(rbf_func[0](lab_i[0], lab_i[1], lab_i[2]), 0, 100),
                    np.clip(rbf_func[1](lab_i[0], lab_i[1], lab_i[2]), -100, 100),
                    np.clip(rbf_func[2](lab_i[0], lab_i[1], lab_i[2]), -100, 100))
    # Compute interpolated values for LUT grid
    interp_table = np.concatenate(np.array(interp_table).T, axis=-1).reshape((len(edges[0]),
                                                                              len(edges[0]),
                                                                              len(edges[0]), 3))

    out_lut = colour.LUT3D(interp_table, 'correction', in_domain, settings.lut_size)    # Create 3D LUT

    return out_lut


def get_angle(start_point, end_point, radians=False, clamp=False):
    """Get angle between 2 points"""
    # Get angle in range [-180, 180]
    angle = math.degrees(math.atan2((end_point[1] - start_point[1]), end_point[0] - start_point[0]))

    if clamp:  # Get angle in range [0, 360]
        if angle < 0:
            angle += 360

    if radians:
        return math.radians(angle)  # Convert angle to radians
    else:
        return angle


def get_key(in_val, in_dict):
    """Get dict key by value"""
    for key in in_dict.keys():
        if in_val == in_dict[key]:
            return key
    return None


def parallel_operation(in_array, const, operation=0, is_thread=False):  # operation - 0: sum, 1: multiply, 2: power
    """Multithreaded calculations"""
    if is_thread or settings.cpu_threads == 0:
        if operation == 0:
            return in_array + const
        elif operation == 1:
            return np.multiply(in_array, const)
        elif operation == 2:
            return np.power(in_array, const)
    else:
        split_array = np.array_split(in_array, min(settings.cpu_threads, 16))   # Split input data into threads

        # Create threads, perform computations
        pool = multiprocessing.pool.ThreadPool()
        out_array = pool.starmap(parallel_operation, zip(split_array, repeat(const), repeat(operation), repeat(True)))

        return np.concatenate(out_array)    # Combine split data


def print_estimate(start_time, progress):
    """Print rough operation time estimate"""
    est_time = round((time.perf_counter() - start_time) * (1 / progress - 1))
    if est_time > 5:
        print()
        if est_time < 60:
            print("Estimated time left:", str(est_time), "s")
        else:
            print("Estimated time left:", str(round(est_time / 60)), "min", str(round(est_time % 60)), "s")
        print()


def yes_no_prompt(prompt):
    """Wait for Y/N answer"""
    while True:
        cur_input = input(f"{prompt} (Y/N): ")
        if cur_input is not None:
            cur_input = cur_input.strip().capitalize()
            if cur_input == 'Y':
                return True
            elif cur_input == 'N':
                return False

        print("Invalid input, only use Y/N.")
