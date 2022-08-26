import settings
import utilities
import main_script
import image_manipulation

import os
import math
import time
from ast import literal_eval as make_tuple
import xml.etree.ElementTree as ElementTree
import xml.dom.minidom as dom
import multiprocessing
import natsort
from itertools import repeat
import numpy as np
import cv2 as cv
import exiftool
from matplotlib import pyplot as plt
import colour

selected_points = ((-1, -1), (-1, -1))
zoom_point = (-1, -1)


# Image event function called upon clicking, etc. on window
def image_event(event, x, y, flags, param):
    global selected_points, zoom_point  # Use global selection variables

    if event == cv.EVENT_LBUTTONDOWN:   # Left mouse button pressed down
        if (param[0] == settings.prompts['horizontal'] or param[0] == settings.prompts['line']
                or param[0] == settings.prompts['crop']):
            if selected_points[1] == (-1, -1):  # 2nd selection point not set
                # Set new point as first selection point, move old to 2nd
                selected_points = ((x, y), selected_points[0])
            elif math.dist((x, y), selected_points[1]) < math.dist((x, y), selected_points[0]):
                # Move closer point
                selected_points = (selected_points[0], (-1, -1))
            else:
                # Move closer point
                selected_points = (selected_points[1], (-1, -1))
        else:
            # Zoom to mouse position
            zoom_point = (x, y)
            image_event(cv.EVENT_MOUSEMOVE, x, y, flags, param)

    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:   # Left/right mouse button released
        if zoom_point != (-1, -1):
            # Adjust zoomed selection points
            sel_coord = np.sum((zoom_point, np.divide(np.subtract((x, y), zoom_point), settings.selection_zoom)),
                               axis=0)
            sel_coord = (round(sel_coord[0]), round(sel_coord[1]))
        else:
            sel_coord = (x, y)

        # Is 2nd selection point closer than 1st?
        check_closer = math.dist(sel_coord, selected_points[1]) < math.dist(sel_coord, selected_points[0])

        if event == cv.EVENT_RBUTTONUP:     # Right mouse button released
            sel_coord = (-1, -1)  # Remove selection
            if (param[0] == settings.prompts['horizontal'] or param[0] == settings.prompts['line']
                    or param[0] == settings.prompts['crop']):
                # Remove both selection points
                selected_points = ((-1, -1), (-1, -1))

        if np.sum(selected_points[0]) == -2:
            selected_points = (selected_points[1], selected_points[0])  # Make sure empty selection is second in list

        if ((sel_coord != (-1, -1) and np.sum(selected_points[1]) == -2)
                or (check_closer and np.sum(selected_points[1]) != -2)):
            selected_points = (sel_coord, selected_points[0])
        else:
            selected_points = (sel_coord, selected_points[1])

        if param[0] == settings.prompts['line']:
            selected_points = (selected_points[1], selected_points[0])

        img_c = (param[1][0].copy(), param[1][1])
        markers = 0
        for i in range(2):
            if selected_points[i] != (-1, -1):
                # Draw cross on selected point
                cv.drawMarker(img_c[0], selected_points[i], (0, 0, main_script.max_val[img_c[1][0][1]]),
                              cv.MARKER_TILTED_CROSS, 15, 2)
                markers += 1
        if markers == 2:    # both selection points exist
            if param[0] == settings.prompts['horizontal']:
                # Draw line between points
                cv.line(img_c[0], selected_points[0], selected_points[1],
                        (0, 0, main_script.max_val[img_c[1][0][1]]), 2)
            elif param[0] == settings.prompts['line']:
                # Draw arrow from 1st to 2nd point
                cv.arrowedLine(img_c[0], selected_points[0], selected_points[1],
                               (0, 0, main_script.max_val[img_c[1][0][1]]), 2,
                               tipLength=(25 / (math.dist(selected_points[0], selected_points[1]) + 1)))
            elif param[0] == settings.prompts['crop']:
                # Draw cropping rectangle
                cv.rectangle(img_c[0], selected_points[0], selected_points[1],
                             (0, 0, main_script.max_val[img_c[1][0][1]]), 2)

        show_image(param[0], img_c, False)

        zoom_point = (-1, -1)   # Clear zoom point

    elif event == cv.EVENT_MOUSEMOVE:   # Mouse position changed
        if (param[0] == settings.prompts['horizontal'] or param[0] == settings.prompts['line']
                or param[0] == settings.prompts['crop']):
            img_c = (param[1][0].copy(), param[1][1])
            if selected_points[0] != (-1, -1) and selected_points[1] == (-1, -1):
                if param[0] == settings.prompts['horizontal']:
                    # Redraw line between points
                    cv.line(img_c[0], selected_points[0], (x, y), (0, 0, main_script.max_val[img_c[1][0][1]] / 2), 2)
                elif param[0] == settings.prompts['line']:
                    # Redraw arrow from 1st to 2nd point
                    cv.arrowedLine(img_c[0], selected_points[0], (x, y),
                                   (0, 0, main_script.max_val[img_c[1][0][1]] / 2), 2,
                                   tipLength=(25 / (math.dist(selected_points[0], (x, y)) + 1)))
                elif param[0] == settings.prompts['crop']:
                    # Redraw cropping rectangle
                    cv.rectangle(img_c[0], selected_points[0], (x, y), (0, 0, main_script.max_val[img_c[1][0][1]] / 2),
                                 2)
                # Redraw selection crosses
                cv.drawMarker(img_c[0], selected_points[0], (0, 0, main_script.max_val[img_c[1][0][1]]),
                              cv.MARKER_TILTED_CROSS, 15, 2)
                show_image(param[0], img_c, False)

        elif zoom_point != (-1, -1):
            # Adjust zoom position
            zoom_coord = np.sum((zoom_point, np.divide(np.subtract((x, y), zoom_point), settings.selection_zoom)),
                                axis=0)
            zoom_coord = (round(zoom_coord[0]), round(zoom_coord[1]))
            img_c = image_manipulation.zoom_image(param[1], settings.selection_zoom, zoom_coord)
            cv.drawMarker(img_c[0], zoom_coord, (0, 0, main_script.max_val[img_c[1][0][1]]),
                          cv.MARKER_TILTED_CROSS, 15, 2)
            show_image(param[0], img_c, False)


# Wait for key input
def wait_key(in_keys=('enter', 'escape', 'space')):  # Wait for any of given keys
    key_dict = {'enter': 13, 'escape': 27, 'space': 32}

    if type(in_keys) is str:
        in_keys = tuple(in_keys)
    while True:
        k = cv.waitKey(0)
        if any(k == key_dict[in_key] for in_key in in_keys):
            return utilities.get_key(k, key_dict)   # Get key by input value


# Get path to given sample
def sample_path(file_name):
    if len(file_name.split('_')[0].split('-')) > 1:
        path = rf"Corrected Images\{file_name.split('_')[0].split('-')[0]}\{file_name.split('_')[0]}"
    else:
        path = rf"Corrected Images\{file_name.split('_')[0]}"

    return path


# Write image file
# Image format: (ndarray, ((color space index, bit depth index), metadata)))
def write_image(in_img, img_name, sub_folder, extension='.' + settings.output_extension,
                show_format=False, convert=True):
    img_path = os.path.join(settings.main_directory, sub_folder)    # Combine path
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if convert:
        if show_format:
            # Write as sRGB 8bit
            conversion = 'show'
            output_color_space = 0
        else:
            # Write with selected color space and bit depth
            conversion = 'out'
            output_color_space = settings.output_color_space
        out_img = image_manipulation.convert_color(in_img, conversion)
    else:
        out_img = in_img
        output_color_space = out_img[1][0][0]

    # Write uncompressed TIFF
    cv.imwrite(os.path.join(img_path, img_name + extension), out_img[0],
               params=(cv.IMWRITE_TIFF_COMPRESSION, 1))

    # Write metadata
    with exiftool.ExifTool("exiftool.exe") as et:
        et.execute(rf"-icc_profile<=ICC Profiles\{settings.color_spaces[output_color_space]}.icc",
                   *[f"-{exif_tag}={out_img[1][1][exif_tag]}" for exif_tag in out_img[1][1].keys()],
                   os.path.join(img_path, img_name + extension))

    # Remove EXIF tool's temporary file
    if os.path.exists(os.path.join(img_path, img_name + extension + "_original")):
        os.remove(os.path.join(img_path, img_name + extension + "_original"))


# Read image file
# Image format: (ndarray, ((color space index, bit depth index), metadata)))
def read_image(img_name, sub_folder='Exported Images', col_depth=-1, convert=True):
    img_path = os.path.join(settings.main_directory, sub_folder, img_name)  # Combine file path
    if str.strip(img_name) == '' or os.path.exists(img_path) is False:
        # Image not found
        return None

    img = cv.imread(img_path, col_depth)    # Read in BGR format

    metadata = [[settings.output_color_space, settings.output_depth], {}]  # Use output settings as default
    with exiftool.ExifToolHelper() as et:
        # Read image metadata
        md = dict(et.get_metadata(img_path)[0])

    # Read image color space
    if 'ICC_Profile:ProfileDescription' in md:
        if settings.color_spaces.count(md['ICC_Profile:ProfileDescription'].split(' ')[0]) == 0:
            utilities.print_color(f"Unsupported color space '{md['ICC_Profile:ProfileDescription']}'!", 'error')
            return None
        else:
            metadata[0][0] = settings.color_spaces.index(md['ICC_Profile:ProfileDescription'].split(' ')[0])
    else:
        utilities.print_color("Color space not found in metadata!", 'error')

    # Read image bit depth
    if 'EXIF:BitsPerSample' in md:
        md_bits = int(str(md['EXIF:BitsPerSample']).split(' ')[0])
    elif 'File:BitsPerSample' in md:
        md_bits = int(str(md['File:BitsPerSample']).split(' ')[0])
    else:
        utilities.print_color("Bit depth not found in metadata!", 'error')
        return None

    if settings.bit_depths.count(md_bits) == 0:
        utilities.print_color("Unsupported bit depth!", 'error')
        return None
    else:
        # Set bit depth
        metadata[0][1] = settings.bit_depths.index(md_bits)

    # Read rest of image metadata
    for data in settings.exif_data:
        if data in md:
            metadata[1][data] = md[data]

    if convert:
        # Convert to LAB D50
        out_img = image_manipulation.convert_color((img, metadata), 'in')
    else:
        # No conversion
        out_img = (img.astype(main_script.bit_type[metadata[0][1]]), metadata)

    return out_img


# Show image in window
def show_image(window_name, in_img, convert=True):
    if convert:
        # Convert to sRGB 8bit
        in_img = image_manipulation.convert_color(in_img, 'show')
    cv.imshow(window_name, in_img[0])
    cv.setWindowProperty(window_name, cv.WND_PROP_TOPMOST, 1)   # Set as top window


# Write correction profile 3D LUT
def write_profile(in_name, in_lut, in_gray, in_metadata, in_ref_data, in_sample_data, ref_grid):
    profile_path = os.path.join(settings.main_directory, r'Calibration\Correction Profiles')    # Combine file path
    if not os.path.exists(profile_path):
        os.makedirs(profile_path)

    # Include image format data
    comments = ['(x, y, z): (L*, a*, b*)', f'gray;{str(in_gray)}',
                [f'ColorSpace;{in_metadata[0][0]}', f'BitDepth;{in_metadata[0][1]}'], [], []]

    # Include image metadata
    for key in in_metadata[1].keys():
        comments[2].append(key + ';' + str(in_metadata[1][key]))
    comments[2] = str(tuple(comments[2]))

    # Include calibration accuracy information
    ref_data_types = ('avg_ciede2000', 'min_ciede2000', 'max_ciede', 'avg_dL', 'avg_da', 'avg_db')
    for i in range(6):
        comments[3].append(ref_data_types[i] + ';' + str(in_ref_data[i]))
    comments[3] = str(tuple(comments[3]))

    # Include color sample information
    sample_data_types = ['ciede2000', 'dL', 'da', 'db']
    for l_y in range(ref_grid[1]):
        for l_x in range(ref_grid[0]):
            sample_text = f'({l_x + 1}, {l_y + 1})'
            for i in range(4):
                sample_text += ';' + sample_data_types[i] + ';' + str(in_sample_data[i, l_x, l_y])
            comments[4].append(sample_text)
    comments[4] = str(tuple(comments[4]))

    # Create final 3D LUT
    out_lut = colour.LUT3D(in_lut.table, in_name, in_lut.domain, in_lut.size,
                           comments)
    print(out_lut)

    # Write 3D LUT as .cube file
    colour.write_LUT(out_lut, os.path.join(profile_path, in_name) + '.cube')

    print(f"Profile '{in_name}' saved.")


# Read calibration profile based on focus height
def read_profile(in_height):  # input height as string
    try:
        if in_height.strip() == '':
            in_height = 10000  # Default to the highest correction profile, usually most accurate
        calib_dist = int(in_height)
        closest = (math.inf, 0)
        for p in os.listdir(os.path.join(settings.main_directory, r'Calibration\Correction Profiles')):
            try:
                # Find the closest focus height profile
                p_dist = int(str.split(p, '.')[0])
                if abs(p_dist - calib_dist) < closest[0]:
                    closest = (abs(p_dist - calib_dist), p_dist)
            except ValueError:
                pass
        in_height = str(closest[1])
    except ValueError:
        pass
    try:
        # Read 3D LUT
        out_lut = colour.read_LUT(os.path.join(settings.main_directory, r'Calibration\Correction Profiles',
                                               in_height + '.cube'))
        print(f"Read correction profile '{in_height}'.")
    except BaseException as error:
        utilities.print_color(f"An exception occurred: {error}", 'error')
        return None

    # Read profile reference gray LAB
    gray_lab = make_tuple(out_lut.comments[1].split(';')[1])

    # Read profile metadata
    metadata = {}
    for md in make_tuple(out_lut.comments[2]):
        metadata[md.split(';')[0]] = md.split(';')[1]

    return out_lut, gray_lab, metadata


# Write sample crop data
def write_crop(ref_points, crop_angle=None, crop_corners=None, gray_labs=None):
    # Combine file path
    crop_path = os.path.join(rf"{settings.main_directory}\Corrected Images",
                             rf"{ref_points[0][0].split('-')[0]}\{ref_points[0][0].split('_')[0]}\Cropped")
    crop_writes = 0

    new_file = True
    if not os.path.exists(crop_path):
        os.makedirs(crop_path)
        root = ElementTree.Element('data')
    elif not os.path.exists(rf"{crop_path}\{ref_points[0][0].split('_')[0]}.xml"):
        root = ElementTree.Element('data')
    else:
        # Previous .xml file found
        root = ElementTree.parse(rf"{crop_path}\{ref_points[0][0].split('_')[0]}.xml").getroot()
        new_file = False

    if gray_labs is None:
        # Writing actual image crop data
        if new_file:
            # Write start ref. data
            start = ElementTree.SubElement(root, 'start')
            start_data = ElementTree.SubElement(start, 'img')
            start_data.attrib['name'] = ref_points[0][0]
            start_data.text = str(ref_points[0][1])

            start_data = ElementTree.SubElement(start, 'angle')
            start_data.text = str(crop_angle)

            start_data = ElementTree.SubElement(start, 'crop')
            start_data.text = str(crop_corners)

            # Create ref. element structure
            ref = ElementTree.SubElement(root, 'references')
        else:
            # Read ref. element structure
            ref = root.find('references')

        # Write/overwrite ref. points
        for i in range(len(ref_points)):
            prev_ref = ref.find(f"img[@name='{ref_points[i][0]}']")
            if not new_file and prev_ref is not None:
                ref.remove(prev_ref)    # Remove previous data
            ref_data = ElementTree.SubElement(ref, 'img')
            ref_data.attrib['name'] = ref_points[i][0]
            ref_data.text = str(ref_points[i][1])
            crop_writes += 1
    else:
        # Writing ref. gray data
        if new_file:
            # Create ref. gray element structure
            grays = ElementTree.SubElement(root, 'grays')
        else:
            # Read ref. gray element structure
            grays = root.find('grays')
        for key in gray_labs:
            # Write/overwrite ref. gray values
            prev_gray = grays.find(f"gray[@measurement='{key}']")
            if not new_file and prev_gray is not None:
                grays.remove(prev_gray)     # Remove previous data
            gray_data = ElementTree.SubElement(grays, 'gray')
            gray_data.attrib['measurement'] = str(key)
            gray_data.text = str(gray_labs[key])
            crop_writes += 1

    # Write .xml file
    with open(rf"{crop_path}\{ref_points[0][0].split('_')[0]}.xml", 'w') as f:
        lines = [line for line in dom.parseString(ElementTree.tostring(root)).toprettyxml().splitlines(True)
                 if line.strip() != '']
        f.writelines(lines)

    print(ref_points[0][0].split('_')[0], f'({crop_writes})', 'crop data written.')


# Read sample crop data
def read_crop(in_name, ref_gray=False):
    # Combine file path
    crop_path = rf"{settings.main_directory}\Corrected Images\{in_name.split('-')[0]}\{in_name.split('_')[0]}\Cropped"

    try:
        # Read crop data .xml file
        root = ElementTree.parse(rf"{crop_path}\{in_name.split('_')[0]}.xml").getroot()
    except FileNotFoundError:
        # Crop data not found
        return None
    except BaseException as error:
        utilities.print_color(f"An exception occurred: {error}", 'error')
        return None

    if ref_gray:
        # Reading ref. gray values
        gray_labs = {}
        if root.find('grays') is not None:
            for measurement in root.find('grays').findall('gray'):
                # Include values in ref. gray dictionary
                gray_labs[int(measurement.attrib['measurement'])] = make_tuple(measurement.text)

        return gray_labs
    else:
        # Reading actual crop data

        # Read start ref. values
        start_points = tuple((root.find('start').find('img').attrib['name'],
                              make_tuple(root.find('start').find('img').text)))
        crop_angle = float(root.find('start').find('angle').text)
        crop_corners = make_tuple(root.find('start').find('crop').text)

        ref_points = {}
        for img in root.find('references').findall('img'):
            # Include points in ref. point dictionary
            ref_points[img.attrib['name']] = make_tuple(img.text)

        return ref_points, start_points, crop_angle, crop_corners


# Read color target data
def read_reference(ref_name):
    try:
        # Read target data .xml file
        root = ElementTree.parse(os.path.join(settings.main_directory, r'Calibration\Reference Values',
                                              ref_name + '.xml')
                                 ).getroot()
        print(f"Read reference '{ref_name}' values.")
    except FileNotFoundError:
        # Target data not found
        return None
    except BaseException as error:
        utilities.print_color(f"An exception occurred: {error}", 'error')
        return None

    # Read color target format
    ref_illuminant = root.find('illuminant').text
    grid = root.find('grid')
    ref_grid = make_tuple(grid.text.strip())
    ref_offset = (make_tuple(grid.find(f"offset[@type='top-left']").text),
                  make_tuple(grid.find(f"offset[@type='spacing']").text),
                  make_tuple(grid.find(f"offset[@type='sample']").text))

    ref_info = (ref_grid, ref_offset)

    # Read color target sample values
    data_types = ('L', 'a', 'b')
    ref_values = np.zeros((ref_grid[0], ref_grid[1], 3))
    for l_y in range(ref_grid[1]):
        for l_x in range(ref_grid[0]):
            for i in range(3):
                ref_values[l_x, l_y, i] = float(root.find('samples').find(f"sample[@grid='({l_x + 1}, {l_y + 1})']")
                                                .find(f"data[@type='{data_types[i]}']").text)
    # Convert ref. data to LAB D50
    if ref_illuminant != 'D50':
        ref_values = image_manipulation.convert_color((ref_values, ref_illuminant), 'adapt')[0]

    return ref_info, ref_values


# Measure series color, mode 0: area, 1: line
def measure_series(path, ref_name, mode, measurement_name):
    global selected_points

    # Read dE ref. image
    ref_img = read_image(ref_name + '.' + settings.output_extension, os.path.join(settings.main_directory, path))

    # List files in sample directory
    file_names = os.listdir(os.path.join(settings.main_directory, path))
    for file_name in file_names:
        if len(str(file_name).split('.')) <= 1 or str(file_name).split('.')[1] != settings.output_extension:
            file_names.remove(file_name)    # Ignore folders

    file_names = natsort.natsorted(file_names)  # Sort files alphabetically

    if mode == 0:   # Measure selected area average color, results in 2D data
        measurement_name = 'area_' + measurement_name

        roi = get_roi(ref_img)[1]   # Select measurement area

        print()
        print("Measuring series", ref_name.split('_')[0], "color ...")

        ref_lab = get_average_color(get_roi(ref_img, 1, in_roi=roi)[0])

        series_data = [('file', 'measurement', 'CIEDE2000', 'L*', 'a*', 'b*', 'C*', 'h°')]  # Data headers
        x = []
        y = []

        # Measure each image of sample
        est_data = [time.perf_counter(), 0]
        for i in range(len(file_names)):
            img = read_image(file_names[i], os.path.join(settings.main_directory, path))
            measurement_roi = get_roi(img, 1, in_roi=roi)
            avg_lab = get_average_color(measurement_roi[0])     # Measure average color of selected area

            # Add measured data to write list
            series_data.append([file_names[i].split('.')[0],  # File name
                                file_names[i].split('.')[0].split('_')[1],  # Measurement no.
                                colour.delta_E(avg_lab, ref_lab, method='CIE 2000'),  # CIEDE2000
                                avg_lab[0],  # L*
                                avg_lab[1],  # a*
                                avg_lab[2],  # b*
                                math.dist((0, 0), (avg_lab[1], avg_lab[2])),  # C*
                                utilities.get_angle((0, 0), (avg_lab[1], avg_lab[2]), clamp=True)  # h°
                                ])
            x.append(file_names[i].split('.')[0].split('_')[1])
            y.append(utilities.get_angle((0, 0), (avg_lab[1], avg_lab[2]), clamp=True))

            # Draw selection rectangle on image
            img_a = image_manipulation.scale_image((cv.rectangle(img[0], (roi[0], roi[1]),
                                                                 (roi[0] + roi[2], roi[1] + roi[3]),
                                                                 (0, 0, main_script.max_val[img[1][0][1]]), 3),
                                                    img[1]))[0]

            # Write image for selection reference
            write_image(img_a,
                        file_names[i].split('.')[0].split('_')[0] + '_' + file_names[i].split('.')[0].split('_')[1],
                        os.path.join(sample_path(file_names[i]), 'Measurements', measurement_name),
                        '_area' + '.' + settings.output_extension, True)

            est_data[1] += 1
            if est_data[1] == 1:
                # Estimate total processing time based on first image
                utilities.print_estimate(est_data[0], est_data[1] / len(file_names))

        # Plot h° measurements
        plt.plot(x, y, 'rx')
        plt.xlabel('measurement')
        plt.ylabel('h°')
        plt.show()

        data_name = file_names[0].split('.')[0].split('_')[0] + '_area_data'

    elif mode == 1:   # Measure pixel values along selected line, results in 3D data
        measurement_name = 'line_' + measurement_name

        while True:
            # Prompt for image pixel scale
            px_scale = input("Pixel scale (µm/px): ")
            try:
                px_scale = float(px_scale.strip())
            except ValueError:
                pass

            if px_scale is not None:
                break
            else:
                print("Invalid scale, only input numbers using '.' as decimal separator.")

        px_scale = px_scale / 1000  # Convert to mm/px
        img_c, img_scale = image_manipulation.scale_image(image_manipulation.convert_color(ref_img, 'show'))
        prompt = settings.prompts['line']

        # Only accept rotation with none (= no rotation or previous rotation if adjusting) or both selection points
        while True:
            show_image(prompt, img_c, False)
            cv.setMouseCallback(prompt, image_event, param=[prompt, img_c])
            key_pressed = wait_key()
            if key_pressed == 'escape':
                # Cancel measurement
                return None
            elif key_pressed == 'enter' and all(np.sum(elem) != -2 for elem in selected_points):
                # Use selected line
                break
        cv.destroyWindow(prompt)
        line_points = np.divide(selected_points, img_scale)     # Scale selection
        selected_points = ((-1, -1), (-1, -1))  # Clear selection

        print()
        print("Measuring series", ref_name.split('_')[0], "color ...")

        # Calculate line length and direction
        line_points = ((round(line_points[0][0]), round(line_points[0][1])),
                       (round(line_points[1][0]), round(line_points[1][1])))
        line_pol = (round(math.dist(line_points[0], line_points[1])),
                    utilities.get_angle(line_points[0], line_points[1]))
        # Measure ref. LAB
        ref_lab = image_manipulation.convert_color((ref_img[0][line_points[0][1]][line_points[0][0]], ref_img[1]),
                                                   'LAB')[0]

        series_data = [('file', 'measurement', 'x', 'CIEDE2000', 'L*', 'a*', 'b*', 'C*', 'h°')]     # Data headers

        est_data = [time.perf_counter(), 0]
        for i in range(len(file_names)):
            img = read_image(file_names[i], os.path.join(settings.main_directory, path))

            series_data.append([file_names[i].split('.')[0],  # File name
                                file_names[i].split('.')[0].split('_')[1],  # Measurement no.
                                [], [], [], [], [], [], []])

            for x in range(line_pol[0] + 1):
                # Measure each point along line
                x_point = np.sum((line_points[0], cvt_point((x, line_pol[1]), -2)), axis=0)
                x_point = (round(x_point[0]), round(x_point[1]))
                x_lab = image_manipulation.convert_color((img[0][x_point[1]][x_point[0]], img[1]), 'LAB')[0]

                series_data[1 + i][2].append(px_scale * x)  # x (dist. along line)
                series_data[1 + i][3].append(colour.delta_E(ref_lab, x_lab, method='CIE 2000'))  # CIEDE 2000
                series_data[1 + i][4].append(x_lab[0])  # L*
                series_data[1 + i][5].append(x_lab[1])  # a*
                series_data[1 + i][6].append(x_lab[2])  # b*
                series_data[1 + i][7].append(math.dist((0, 0), (x_lab[1], x_lab[2])))  # C*
                series_data[1 + i][8].append(utilities.get_angle((0, 0), (x_lab[1], x_lab[2]), clamp=True))  # h°

            # Plot h° along line
            plt.plot(series_data[1 + i][2], series_data[1 + i][8], 'rx')
            plt.xlabel('x (mm)')
            plt.ylabel('h°')
            # plt.ylim((0, 50))     # Set y-axis range
            plt.show()

            # Draw selected line on image
            img_l = image_manipulation.scale_image(
                (cv.arrowedLine(img[0], line_points[0], line_points[1],
                                (0, 0, main_script.max_val[img[1][0][1]]), round(2 / img_scale),
                                tipLength=(25 / math.dist(line_points[0], line_points[1]) / img_scale)),
                 img[1]))[0]

            # Write image for selection reference
            write_image(img_l,
                        file_names[i].split('.')[0].split('_')[0] + '_' + file_names[i].split('.')[0].split('_')[1],
                        os.path.join(sample_path(file_names[i]), 'Measurements', measurement_name),
                        '_line' + '.' + settings.output_extension, True)

            est_data[1] += 1
            if est_data[1] == 1:
                # Estimate total processing time based on first image
                utilities.print_estimate(est_data[0], est_data[1] / len(file_names))

        data_name = file_names[0].split('.')[0].split('_')[0] + '_line_data'
    else:
        print("Invalid measuring mode.")
        return

    # Write measurement data to .csv file
    utilities.write_csv(series_data, data_name,
                        os.path.join(sample_path(file_names[0]), 'Measurements', measurement_name))


# Get image ROI (selected area)
# Mode 0: Select & output, 1: Apply & output
def get_roi(in_img, mode=0, in_roi=(0, 0, 0, 0), show_format=False):
    global selected_points  # Use global selection variables
    prompt = settings.prompts['crop']
    roi = in_roi
    out_roi = None
    key_pressed = None

    if mode == 0:   # Select & output
        if show_format:
            img_s = in_img  # No conversion
        else:
            img_s = image_manipulation.convert_color(in_img, 'show')    # Convert image to sRGB 8bit
        img_s, img_scale = image_manipulation.scale_image(img_s)

        if np.sum(in_roi) != 0:
            # Round selection values
            selected_points = ([round(sel_point * img_scale) for sel_point in (in_roi[0], in_roi[1])],
                               [round(sel_point * img_scale) for sel_point in
                                (in_roi[0] + in_roi[2], in_roi[1] + in_roi[3])])

        # Only accept reference with both selection points
        while key_pressed is None or any(np.sum(elem) == -2 for elem in selected_points):
            show_image(prompt, img_s, False)
            cv.setMouseCallback(prompt, image_event, param=[prompt, img_s])
            image_event(cv.EVENT_LBUTTONUP,
                        selected_points[0][0], selected_points[0][1],
                        None, [prompt, img_s])
            key_pressed = wait_key()
            if key_pressed == 'escape':
                # Cancel cropping
                return None
            elif key_pressed == 'space':
                # Use defaults
                break
        cv.destroyWindow(prompt)

        if key_pressed == 'enter':
            ref_points = np.divide(selected_points,
                                   img_scale)
            # Make sure first selection points is on the left
            if ref_points[0][0] > ref_points[1][0]:
                ref_points = (ref_points[1], ref_points[0])

            # Make sure first selection point is on top
            if ref_points[0][1] > ref_points[1][1]:
                ref_points = ((ref_points[0][0], ref_points[1][1]), (ref_points[1][0], ref_points[0][1]))

            selected_points = ((-1, -1), (-1, -1))  # Clear selection

            # Convert selection to ROI format
            roi = (ref_points[0][0], ref_points[0][1],
                   abs(ref_points[1][0] - ref_points[0][0]), abs(ref_points[1][1] - ref_points[0][1]))
            roi = tuple(round(coord) for coord in roi)
            out_roi = roi

    # Default to full image
    if np.sum(roi) == 0 or key_pressed == 'space':
        out_roi = None
        roi = (0, 0, in_img[0].shape[1], in_img[0].shape[0])

    return ((in_img[0][int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])], in_img[1]),
            out_roi, key_pressed)


# Crop image to safety margins
def get_safe_area(in_img):
    return (in_img[0][round(in_img[0].shape[0] * settings.ref_margins[1]):
                      round(in_img[0].shape[0] * (1 - settings.ref_margins[1])),
            round(in_img[0].shape[1] * settings.ref_margins[0]):
            round(in_img[0].shape[1] * (1 - settings.ref_margins[0]))],
            in_img[1])


# Get image average color
def get_average_color(in_img):  # Return average color of image
    return np.average(np.average(in_img[0], axis=0), axis=0)


# Mode -2: polar->relative, -1: relative->corner, 1: corner->relative, 2: relative->polar
def cvt_point(in_coords, mode, img_shape=(-1, -1, -1)):
    if mode == -2:
        # Coordinate format: (radius, degrees) -> (x, y)
        out_coords = (math.cos(math.radians(in_coords[1])) * in_coords[0],
                      math.sin(math.radians(in_coords[1])) * in_coords[0])
    elif mode == -1:
        # Coordinate format: (x, y) -> (x, y)
        out_coords = (round(img_shape[1] / 2 + in_coords[0]), round(img_shape[0] / 2 - in_coords[1]))
    elif mode == 1:
        # Coordinate format: (x, y) -> (x, y)
        out_coords = (round(in_coords[0] - img_shape[1] / 2), round(img_shape[0] / 2 - in_coords[1]))
    elif mode == 2:
        # Coordinate format: (x, y) -> (radius, degrees)
        out_coords = (math.dist((0, 0), in_coords), utilities.get_angle((0, 0), in_coords, clamp=True))
    else:
        utilities.print_color("Invalid point conversion mode.", 'error')
        return None

    return out_coords


# Check if capture settings match
def compare_settings(img_metadata, ref_metadata):
    # List of capture settings to consider
    settings_list = ['EXIF:Make', 'EXIF:Model', 'EXIF:FNumber', 'EXIF:ExposureTime', 'EXIF:ISO', 'EXIF:FocalLength']
    for setting in settings_list:  # Loop through settings
        if setting in img_metadata and setting in ref_metadata:
            if str(img_metadata[setting]) != str(ref_metadata[setting]):
                return False  # Capture settings don't match

    return True  # Capture settings match


# Perform function in parallel threads
def parallel_process(function, in_img, params, operations=settings.cpu_threads):
    if type(params[-1]) is int and 0 <= params[-1]:
        operations = params[-1]
    split_img = np.array_split(in_img[0], operations)   # Split image for threads
    for i in range(len(split_img)):
        split_img[i] = (split_img[i], in_img[1])    # Include metadata in split image

    # Perform parallel processes
    pool = multiprocessing.pool.ThreadPool()
    pool_res = pool.starmap(function, zip(split_img, *[repeat(param) for param in params]))

    # Combine split images
    out_img = []
    for res in pool_res:
        out_img.append(res[0])
    out_img = (np.concatenate(out_img), in_img[1])

    return out_img
