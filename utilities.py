import math
import os
from ast import literal_eval as make_tuple
import xml.etree.ElementTree as ElementTree
import xml.dom.minidom as dom
import multiprocessing
from itertools import repeat
import numpy as np
import cv2 as cv  # Import OpenCV library

import settings
import main_script
import image_manipulation

selected_points = ((-1, -1), (-1, -1))
zoom_point = (-1, -1)


def create_directories(main_dir):
    folders = (r'..\Images', r'Calibration\Image Uniformity', r'Calibration\Correction Profiles',
               r'Calibration\Spectra', r'Calibration\Calibration Images', r'Calibration\Reference Values',
               'Corrected Images', 'Exported Images', 'Remote Capture')
    for folder in folders:
        path = os.path.join(main_dir, folder)
        if not os.path.exists(path):
            os.makedirs(path)


def read_image(img_name, sub_folder='Exported Images', col_depth=-1):
    img_path = os.path.join(settings.main_directory, sub_folder, img_name)
    if str.strip(img_name) == '' or os.path.exists(img_path) is False:
        return None

    img = cv.imread(img_path, col_depth)

    return img


def write_image(in_img, img_name, extension='.' + settings.output_extension, sub_folder='Corrected Images'):
    img_path = os.path.join(settings.main_directory, sub_folder)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if settings.output_colour_space != settings.input_colour_space:
        in_img = image_manipulation.convert_colour(in_img, 3)
    cv.imwrite(os.path.join(img_path, img_name + extension), in_img,
               params=(cv.IMWRITE_TIFF_COMPRESSION, 1))
    # TODO: Include colour space in EXIF for proper viewing


def read_profile(in_name):
    try:
        calib_dist = int(in_name)
        closest = (1000, 0)
        for p in os.listdir(os.path.join(settings.main_directory, r'Calibration\Correction Profiles')):
            try:
                p_dist = int(str.split(p, '.')[0])
                if abs(p_dist - calib_dist) < closest[0]:
                    closest = (abs(p_dist - calib_dist), p_dist)
            except:
                pass
        in_name = str(closest[1])
    except:
        pass
    try:
        root = ElementTree.parse(os.path.join(settings.main_directory, r'Calibration\Correction Profiles',
                                              in_name + '.xml')
                                 ).getroot()
        print(f"Read profile '{in_name}'.")
    except:
        return None

    out_fits = [[], [], []]
    col_types = ('L', 'a', 'b')
    for i in range(3):
        for o in range(4):
            out_fits[i].append(float(root.find('fit').find(f"colour[@type='{col_types[i]}']")
                                     .find(f"term[@degree='{o + 1}']").text))

    return out_fits


def write_profile(in_name, in_fits, in_ref_data, in_sample_data, ref_grid):
    profile_path = os.path.join(settings.main_directory, r'Calibration\Correction Profiles')
    if not os.path.exists(profile_path):
        os.makedirs(profile_path)

    root = ElementTree.Element('data')
    fit = ElementTree.SubElement(root, 'fit')

    elements = ('L', 'a', 'b')
    for i in range(3):
        elem = ElementTree.SubElement(fit, 'colour')
        elem.attrib['type'] = elements[i]
        for o in range(4):
            term = ElementTree.SubElement(elem, 'term')
            term.attrib['degree'] = str(4 - o)
            term.text = str(in_fits[i][o])
    deltas = ElementTree.SubElement(root, 'deltas')
    ref_data_types = ('avg ciede2000', 'min ciede2000', 'max ciede', 'avg dL', 'avg da', 'avg db')
    for i in range(6):
        delta = ElementTree.SubElement(deltas, 'delta')
        delta.attrib['type'] = ref_data_types[i]
        delta.text = str(in_ref_data[i])

    delta = ElementTree.SubElement(deltas, 'delta')
    delta.attrib['type'] = 'samples'
    for l_y in range(ref_grid[1]):
        for l_x in range(ref_grid[0]):
            sample = ElementTree.SubElement(delta, 'sample')
            sample.attrib['grid'] = f'({l_x + 1}, {l_y + 1})'
            sample_data_types = ['ciede2000', 'dL', 'da', 'db']
            for i in range(4):
                sample_data = ElementTree.SubElement(sample, 'data')
                sample_data.attrib['type'] = sample_data_types[i]
                sample_data.text = str(in_sample_data[i, l_x, l_y])

    with open(os.path.join(profile_path, in_name + '.xml'), 'w') as f:
        f.write(dom.parseString(ElementTree.tostring(root)).toprettyxml())


def write_crop(ref_points, crop_angle=None, crop_corners=None):
    crop_path = os.path.join(rf"{settings.main_directory}\Corrected Images",
                             rf"{ref_points[0][0].split('-')[0]}\{ref_points[0][0].split('_')[0]}\Cropped")

    new_file = True
    if not os.path.exists(crop_path):
        os.makedirs(crop_path)
        root = ElementTree.Element('data')
    elif not os.path.exists(rf"{crop_path}\{ref_points[0][0].split('_')[0]}.xml"):
        root = ElementTree.Element('data')
    else:
        root = ElementTree.parse(rf"{crop_path}\{ref_points[0][0].split('_')[0]}.xml").getroot()
        new_file = False

    if crop_angle is not None:
        if not new_file:
            root.remove(root.find('start'))
        start = ElementTree.SubElement(root, 'start')
        start_data = ElementTree.SubElement(start, 'img')
        start_data.attrib['name'] = ref_points[0][0]
        start_data.text = str(ref_points[0][1])

        start_data = ElementTree.SubElement(start, 'angle')
        start_data.text = str(crop_angle)

        start_data = ElementTree.SubElement(start, 'crop')
        start_data.text = str(crop_corners)

    if new_file:
        ref = ElementTree.SubElement(root, 'references')
    else:
        ref = root.find('references')
    for i in range(len(ref_points)):
        prev_ref = ref.find(f"img[@name='{ref_points[i][0]}']")
        if not new_file and prev_ref is not None:
            ref.remove(prev_ref)
        ref_data = ElementTree.SubElement(ref, 'img')
        ref_data.attrib['name'] = ref_points[i][0]
        ref_data.text = str(ref_points[i][1])

    with open(rf"{crop_path}\{ref_points[0][0].split('_')[0]}.xml", 'w') as f:
        lines = [line for line in dom.parseString(ElementTree.tostring(root)).toprettyxml().splitlines(True)
                 if line.strip() != '']
        f.writelines(lines)
    print(ref_points[0][0].split('_')[0], f'({len(ref_points)})', 'crop data written.')


def read_crop(in_name):
    crop_path = rf"{settings.main_directory}\Corrected Images\{in_name.split('-')[0]}\{in_name.split('_')[0]}\Cropped"

    try:
        root = ElementTree.parse(rf"{crop_path}\{in_name.split('_')[0]}.xml").getroot()
    except:
        return None

    start_points = tuple((root.find('start').find('img').attrib['name'],
                          make_tuple(root.find('start').find('img').text)))
    crop_angle = float(root.find('start').find('angle').text)
    crop_corners = make_tuple(root.find('start').find('crop').text)
    ref_points = {}
    for img in root.find('references').findall('img'):
        ref_points[img.attrib['name']] = make_tuple(img.text)

    return ref_points, start_points, crop_angle, crop_corners


def read_reference(ref_name='colorchecker'):
    try:
        root = ElementTree.parse(os.path.join(settings.main_directory, r'Calibration\Reference Values',
                                              ref_name + '.xml')
                                 ).getroot()
        print(f"Read reference '{ref_name}' values.")
    except:
        return None, None, None

    ref_illuminant = root.find('illuminant').text
    ref_grid = make_tuple(root.find('grid').text)

    data_types = ('L', 'a', 'b')
    ref_values = np.zeros((ref_grid[0], ref_grid[1], 3))
    for l_y in range(ref_grid[1]):
        for l_x in range(ref_grid[0]):
            for i in range(3):
                ref_values[l_x, l_y, i] = float(root.find('samples').find(f"sample[@grid='({l_x + 1}, {l_y + 1})']")
                                                .find(f"data[@type='{data_types[i]}']").text)

    return ref_illuminant, ref_grid, ref_values


def image_event(event, x, y, flags, param):
    global selected_points, zoom_point

    if event == cv.EVENT_LBUTTONDOWN:
        if param[0] == "Drag horizontal line, then press enter.":
            if selected_points[1] == (-1, -1):
                selected_points = ((x, y), selected_points[0])
            elif math.dist((x, y), selected_points[1]) < math.dist((x, y), selected_points[0]):
                selected_points = (selected_points[0], (-1, -1))
            else:
                selected_points = (selected_points[1], (-1, -1))
        else:
            zoom_point = (x, y)
            image_event(cv.EVENT_MOUSEMOVE, x, y, flags, param)
    elif event == cv.EVENT_LBUTTONUP:
        if zoom_point != (-1, -1):
            sel_coord = np.sum((zoom_point, np.divide(np.subtract((x, y), zoom_point), settings.selection_zoom)),
                               axis=0)
            sel_coord = (round(sel_coord[0]), round(sel_coord[1]))
        else:
            sel_coord = (x, y)
        if (selected_points[1] == (-1, -1) or
                math.dist(sel_coord, selected_points[1]) < math.dist(sel_coord, selected_points[0])):
            selected_points = (sel_coord, selected_points[0])
        else:
            selected_points = (sel_coord, selected_points[1])

        img_c = param[1].copy()
        markers = 0
        for i in range(2):
            if selected_points[i] != (-1, -1):
                cv.drawMarker(img_c, selected_points[i], (0, 0, main_script.max_val[settings.input_depth]),
                              cv.MARKER_TILTED_CROSS, 15, 2)
                markers += 1
        if markers == 2 and param[0] == "Drag horizontal line, then press enter.":
            cv.line(img_c, selected_points[0], selected_points[1],
                    (0, 0, main_script.max_val[settings.input_depth]), 2)

        cv.imshow(param[0], img_c)

        zoom_point = (-1, -1)

    elif event == cv.EVENT_MOUSEMOVE:
        if param[0] == "Drag horizontal line, then press enter.":
            img_c = param[1].copy()
            if selected_points[0] != (-1, -1) and selected_points[1] == (-1, -1):
                cv.line(img_c, selected_points[0], (x, y), (0, 0, main_script.max_val[settings.input_depth] / 2), 2)
                cv.drawMarker(img_c, selected_points[0], (0, 0, main_script.max_val[settings.input_depth]),
                              cv.MARKER_TILTED_CROSS, 15, 2)
                cv.imshow(param[0], img_c)

        elif zoom_point != (-1, -1):
            zoom_coord = np.sum((zoom_point, np.divide(np.subtract((x, y), zoom_point), settings.selection_zoom)),
                                axis=0)
            zoom_coord = (round(zoom_coord[0]), round(zoom_coord[1]))
            img_c = image_manipulation.zoom_image(param[1], settings.selection_zoom, zoom_coord)
            cv.drawMarker(img_c, zoom_coord, (0, 0, main_script.max_val[settings.input_depth]),
                          cv.MARKER_TILTED_CROSS, 15, 2)
            cv.imshow(param[0], img_c)


def get_angle(start_point, end_point, radians=False):  # Get angle between 2 points in range [-90, 90]
    if end_point[0] - start_point[0] != 0:
        angle = math.atan2(-(end_point[1] - start_point[1]), end_point[0] - start_point[0])
    else:
        angle = 90

    # Clamp to range [-360, 360] (unused)
    if angle < -360 or angle > 360:
        angle = angle + int(-angle / 360) * 360

    # Clamp to range [-180, 180]
    if angle < -180:
        angle = 360 + angle
    elif angle > 180:
        angle = angle - 360

    # Clamp to range [-90, 90]
    if angle < -90:
        angle += 180
    elif angle > 90:
        angle -= 180

    if radians:
        return angle
    else:
        return math.degrees(angle)


# Mode 0: Select & output, 1: Apply & output
def get_roi(in_img, mode=0, in_prompt="Select area, then press ENTER", in_roi=(0, 0, 0, 0)):
    if mode == 0:
        img_s, img_scale = image_manipulation.scale_image(in_img)
        roi = cv.selectROI(in_prompt, img_s, False, False)
        cv.destroyWindow(in_prompt)

        roi = tuple(round(coord / img_scale) for coord in roi)
    else:
        roi = in_roi

    if roi[2] > 0 and roi[3] > 0:
        return in_img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])], roi
    else:
        return in_img.copy(), roi


def get_average_colour(in_img):  # Return average colour of image as (LAB, RGB)
    avg_col = np.average(np.average(in_img, axis=0), axis=0)
    return image_manipulation.convert_colour(avg_col), image_manipulation.convert_colour(avg_col, 0)


# Mode -2: polar->relative, -1: relative->corner, 1: corner->relative, 2: relative->polar
def cvt_point(in_coords, mode, img_shape=(-1, -1, -1)):
    if mode == -2:
        out_coords = (math.cos(math.radians(in_coords[1])) * in_coords[0],
                      -math.sin(math.radians(in_coords[1])) * in_coords[0])
    elif mode == -1:
        out_coords = (round(img_shape[1] / 2 + in_coords[0]), round(img_shape[0] / 2 - in_coords[1]))
    elif mode == 1:
        out_coords = (round(in_coords[0] - img_shape[1] / 2), round(img_shape[0] / 2 - in_coords[1]))
    elif mode == 2:
        out_coords = (math.dist((0, 0), in_coords), get_angle((0, 0), in_coords))
    else:
        print("Invalid point conversion mode.")
        return None

    return out_coords


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
=======
# TODO: Comments
import math
import os
from ast import literal_eval as make_tuple
import xml.etree.ElementTree as ElementTree
import xml.dom.minidom as dom
import multiprocessing
from itertools import repeat
import numpy as np
import cv2 as cv  # Import OpenCV library

import settings
import main_script
import image_manipulation

selected_points = ((-1, -1), (-1, -1))
zoom_point = (-1, -1)


def create_directories(main_dir):
    folders = (r'..\Images', r'Calibration\Image Uniformity', r'Calibration\Correction Profiles',
               r'Calibration\ICC Profiles', r'Calibration\Calibration Images', 'Corrected Images', 'Exported Images',
               'Remote Capture')
    for folder in folders:
        path = os.path.join(main_dir, folder)
        if not os.path.exists(path):
            os.makedirs(path)


def read_image(img_name, sub_folder='Exported Images', col_depth=-1):
    img_path = os.path.join(settings.main_directory, sub_folder, img_name)
    if str.strip(img_name) == '' or os.path.exists(img_path) is False:
        return None

    img = cv.imread(img_path, col_depth)

    return img


def write_image(in_img, img_name, extension='.' + settings.output_extension, sub_folder='Corrected Images'):
    img_path = os.path.join(settings.main_directory, sub_folder)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if settings.output_colour_space != settings.input_colour_space:
        in_img = image_manipulation.convert_colour(in_img, 3)
    cv.imwrite(os.path.join(img_path, img_name + extension), in_img,
               params=(cv.IMWRITE_TIFF_COMPRESSION, 1))
    # TODO: Include colour space in EXIF for proper viewing


def read_profile(in_name):
    try:
        calib_dist = int(in_name)
        closest = (1000, 0)
        for p in os.listdir(os.path.join(settings.main_directory, r'Calibration\Correction Profiles')):
            try:
                p_dist = int(str.split(p, '.')[0])
                if abs(p_dist - calib_dist) < closest[0]:
                    closest = (abs(p_dist - calib_dist), p_dist)
            except:
                pass
        in_name = str(closest[1])
    except:
        pass
    try:
        root = ElementTree.parse(os.path.join(settings.main_directory, r'Calibration\Correction Profiles',
                                              in_name + '.xml')
                                 ).getroot()
        print(f"Read profile '{in_name}'.")
    except:
        return None

    out_fits = [[], [], []]
    col_types = ('L', 'a', 'b')
    for i in range(3):
        for o in range(4):
            out_fits[i].append(float(root.find('fit').find(f"colour[@type='{col_types[i]}']")
                                     .find(f"term[@degree='{o + 1}']").text))

    return out_fits


def write_profile(in_name, in_fits, in_ref_data, in_sample_data, ref_grid):
    profile_path = os.path.join(settings.main_directory, r'Calibration\Correction Profiles')
    if not os.path.exists(profile_path):
        os.makedirs(profile_path)

    root = ElementTree.Element('data')
    fit = ElementTree.SubElement(root, 'fit')

    elements = ('L', 'a', 'b')
    for i in range(3):
        elem = ElementTree.SubElement(fit, 'colour')
        elem.attrib['type'] = elements[i]
        for o in range(4):
            term = ElementTree.SubElement(elem, 'term')
            term.attrib['degree'] = str(4 - o)
            term.text = str(in_fits[i][o])
    deltas = ElementTree.SubElement(root, 'deltas')
    ref_data_types = ('avg ciede2000', 'min ciede2000', 'max ciede', 'avg dL', 'avg da', 'avg db')
    for i in range(6):
        delta = ElementTree.SubElement(deltas, 'delta')
        delta.attrib['type'] = ref_data_types[i]
        delta.text = str(in_ref_data[i])

    delta = ElementTree.SubElement(deltas, 'delta')
    delta.attrib['type'] = 'samples'
    for l_y in range(ref_grid[1]):
        for l_x in range(ref_grid[0]):
            sample = ElementTree.SubElement(delta, 'sample')
            sample.attrib['grid'] = f'({l_x + 1}, {l_y + 1})'
            sample_data_types = ['ciede2000', 'dL', 'da', 'db']
            for i in range(4):
                sample_data = ElementTree.SubElement(sample, 'data')
                sample_data.attrib['type'] = sample_data_types[i]
                sample_data.text = str(in_sample_data[i, l_x, l_y])

    with open(os.path.join(profile_path, in_name + '.xml'), 'w') as f:
        f.write(dom.parseString(ElementTree.tostring(root)).toprettyxml())


def write_crop(ref_points, crop_angle=None, crop_corners=None):
    crop_path = os.path.join(rf"{settings.main_directory}\Corrected Images",
                             rf"{ref_points[0][0].split('-')[0]}\{ref_points[0][0].split('_')[0]}\Cropped")

    new_file = True
    if not os.path.exists(crop_path):
        os.makedirs(crop_path)
        root = ElementTree.Element('data')
    elif not os.path.exists(rf"{crop_path}\{ref_points[0][0].split('_')[0]}.xml"):
        root = ElementTree.Element('data')
    else:
        root = ElementTree.parse(rf"{crop_path}\{ref_points[0][0].split('_')[0]}.xml").getroot()
        new_file = False

    if crop_angle is not None:
        if not new_file:
            root.remove(root.find('start'))
        start = ElementTree.SubElement(root, 'start')
        start_data = ElementTree.SubElement(start, 'img')
        start_data.attrib['name'] = ref_points[0][0]
        start_data.text = str(ref_points[0][1])

        start_data = ElementTree.SubElement(start, 'angle')
        start_data.text = str(crop_angle)

        start_data = ElementTree.SubElement(start, 'crop')
        start_data.text = str(crop_corners)

    if new_file:
        ref = ElementTree.SubElement(root, 'references')
    else:
        ref = root.find('references')
    for i in range(len(ref_points)):
        prev_ref = ref.find(f"img[@name='{ref_points[i][0]}']")
        if not new_file and prev_ref is not None:
            ref.remove(prev_ref)
        ref_data = ElementTree.SubElement(ref, 'img')
        ref_data.attrib['name'] = ref_points[i][0]
        ref_data.text = str(ref_points[i][1])

    with open(rf"{crop_path}\{ref_points[0][0].split('_')[0]}.xml", 'w') as f:
        lines = [line for line in dom.parseString(ElementTree.tostring(root)).toprettyxml().splitlines(True)
                 if line.strip() != '']
        f.writelines(lines)
    print(ref_points[0][0].split('_')[0], f'({len(ref_points)})', 'crop data written.')


def read_crop(in_name):
    crop_path = rf"{settings.main_directory}\Corrected Images\{in_name.split('-')[0]}\{in_name.split('_')[0]}\Cropped"

    try:
        root = ElementTree.parse(rf"{crop_path}\{in_name.split('_')[0]}.xml").getroot()
    except:
        return None

    start_points = tuple((root.find('start').find('img').attrib['name'],
                          make_tuple(root.find('start').find('img').text)))
    crop_angle = float(root.find('start').find('angle').text)
    crop_corners = make_tuple(root.find('start').find('crop').text)
    ref_points = {}
    for img in root.find('references').findall('img'):
        ref_points[img.attrib['name']] = make_tuple(img.text)

    return ref_points, start_points, crop_angle, crop_corners


def read_reference(ref_name='colorchecker'):
    try:
        root = ElementTree.parse(os.path.join(settings.main_directory, r'Calibration\Reference Values',
                                              ref_name + '.xml')
                                 ).getroot()
        print(f"Read reference '{ref_name}' values.")
    except:
        return None, None, None

    ref_illuminant = root.find('illuminant').text
    ref_grid = make_tuple(root.find('grid').text)

    data_types = ('L', 'a', 'b')
    ref_values = np.zeros((ref_grid[0], ref_grid[1], 3))
    for l_y in range(ref_grid[1]):
        for l_x in range(ref_grid[0]):
            for i in range(3):
                ref_values[l_x, l_y, i] = float(root.find('samples').find(f"sample[@grid='({l_x + 1}, {l_y + 1})']")
                                                .find(f"data[@type='{data_types[i]}']").text)

    return ref_illuminant, ref_grid, ref_values


def image_event(event, x, y, flags, param):
    global selected_points, zoom_point

    if event == cv.EVENT_LBUTTONDOWN:
        if param[0] == "Drag horizontal line, then press enter.":
            if selected_points[1] == (-1, -1):
                selected_points = ((x, y), selected_points[0])
            elif math.dist((x, y), selected_points[1]) < math.dist((x, y), selected_points[0]):
                selected_points = (selected_points[0], (-1, -1))
            else:
                selected_points = (selected_points[1], (-1, -1))
        else:
            zoom_point = (x, y)
            image_event(cv.EVENT_MOUSEMOVE, x, y, flags, param)
    elif event == cv.EVENT_LBUTTONUP:
        if zoom_point != (-1, -1):
            sel_coord = np.sum((zoom_point, np.divide(np.subtract((x, y), zoom_point), settings.selection_zoom)),
                               axis=0)
            sel_coord = (round(sel_coord[0]), round(sel_coord[1]))
        else:
            sel_coord = (x, y)
        if (selected_points[1] == (-1, -1) or
                math.dist(sel_coord, selected_points[1]) < math.dist(sel_coord, selected_points[0])):
            selected_points = (sel_coord, selected_points[0])
        else:
            selected_points = (sel_coord, selected_points[1])

        img_c = param[1].copy()
        markers = 0
        for i in range(2):
            if selected_points[i] != (-1, -1):
                cv.drawMarker(img_c, selected_points[i], (0, 0, main_script.max_val[settings.input_depth]),
                              cv.MARKER_TILTED_CROSS, 15, 2)
                markers += 1
        if markers == 2 and param[0] == "Drag horizontal line, then press enter.":
            cv.line(img_c, selected_points[0], selected_points[1],
                    (0, 0, main_script.max_val[settings.input_depth]), 2)

        cv.imshow(param[0], img_c)

        zoom_point = (-1, -1)

    elif event == cv.EVENT_MOUSEMOVE:
        if param[0] == "Drag horizontal line, then press enter.":
            img_c = param[1].copy()
            if selected_points[0] != (-1, -1) and selected_points[1] == (-1, -1):
                cv.line(img_c, selected_points[0], (x, y), (0, 0, main_script.max_val[settings.input_depth] / 2), 2)
                cv.drawMarker(img_c, selected_points[0], (0, 0, main_script.max_val[settings.input_depth]),
                              cv.MARKER_TILTED_CROSS, 15, 2)
                cv.imshow(param[0], img_c)

        elif zoom_point != (-1, -1):
            zoom_coord = np.sum((zoom_point, np.divide(np.subtract((x, y), zoom_point), settings.selection_zoom)),
                                axis=0)
            zoom_coord = (round(zoom_coord[0]), round(zoom_coord[1]))
            img_c = image_manipulation.zoom_image(param[1], settings.selection_zoom, zoom_coord)
            cv.drawMarker(img_c, zoom_coord, (0, 0, main_script.max_val[settings.input_depth]),
                          cv.MARKER_TILTED_CROSS, 15, 2)
            cv.imshow(param[0], img_c)


def get_angle(start_point, end_point, radians=False):  # Get angle between 2 points in range [-90, 90]
    if end_point[0] - start_point[0] != 0:
        angle = math.atan2(-(end_point[1] - start_point[1]), end_point[0] - start_point[0])
    else:
        angle = 90

    # Clamp to range [-360, 360] (unused)
    if angle < -360 or angle > 360:
        angle = angle + int(-angle / 360) * 360

    # Clamp to range [-180, 180]
    if angle < -180:
        angle = 360 + angle
    elif angle > 180:
        angle = angle - 360

    # Clamp to range [-90, 90]
    if angle < -90:
        angle += 180
    elif angle > 90:
        angle -= 180

    if radians:
        return angle
    else:
        return math.degrees(angle)


# Mode 0: Select & output, 1: Apply & output
def get_roi(in_img, mode=0, in_prompt="Select area, then press ENTER", in_roi=(0, 0, 0, 0)):
    if mode == 0:
        img_s, img_scale = image_manipulation.scale_image(in_img)
        roi = cv.selectROI(in_prompt, img_s, False, False)
        cv.destroyWindow(in_prompt)

        roi = tuple(round(coord / img_scale) for coord in roi)
    else:
        roi = in_roi

    if roi[2] > 0 and roi[3] > 0:
        return in_img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])], roi
    else:
        return in_img.copy(), roi


def get_average_colour(in_img):  # Return average colour of image as (LAB, RGB)
    avg_col = np.average(np.average(in_img, axis=0), axis=0)
    return image_manipulation.convert_colour(avg_col), image_manipulation.convert_colour(avg_col, 0)


# Mode -2: polar->relative, -1: relative->corner, 1: corner->relative, 2: relative->polar
def cvt_point(in_coords, mode, img_shape=(-1, -1, -1)):
    if mode == -2:
        out_coords = (math.cos(math.radians(in_coords[1])) * in_coords[0],
                      -math.sin(math.radians(in_coords[1])) * in_coords[0])
    elif mode == -1:
        out_coords = (round(img_shape[1] / 2 + in_coords[0]), round(img_shape[0] / 2 - in_coords[1]))
    elif mode == 1:
        out_coords = (round(in_coords[0] - img_shape[1] / 2), round(img_shape[0] / 2 - in_coords[1]))
    elif mode == 2:
        out_coords = (math.dist((0, 0), in_coords), get_angle((0, 0), in_coords))
    else:
        print("Invalid point conversion mode.")
        return None

    return out_coords


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
>>>>>>> c3c5e1c (Time estimates for long loop operations.)
