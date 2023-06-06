import settings
import utilities
import image_utilities
import main_script

import os
import math
import numpy as np
import cv2 as cv
import colour
import natsort


# Conversion: 'in': BGR(input)->LAB(D50) 'out': LAB(D50)->BGR(output) 'show': LAB(D50)->BGR(0-255 8bit sRGB)
#             'adapt': LAB->LAB(D50) 'LAB': LAB(D50)->LAB(output) 'RGB': LAB(D50)->RGB(0.0-1.0 sRGB) 'xy': LAB(D50)->xy
def convert_color(in_cols, conversion, is_thread=False, operations=1):
    # If already thread or no need for threading, directly compute
    if is_thread or settings.cpu_threads == 0 or max(in_cols[0].shape) < settings.cpu_threads:
        # Split data based on memory usage
        if len(in_cols[0].shape) > 1:
            memory_usage = np.prod(in_cols[0].shape) * operations
            split_cols = np.array_split(in_cols[0], math.ceil(memory_usage / settings.system_memory))
        else:
            split_cols = [in_cols[0]]

        out_cols = []
        for i in range(len(split_cols)):
            split_cols[i] = (split_cols[i], in_cols[1])  # Include metadata in split colors

            if conversion == 'in':  # BGR(input) -> LAB(D50)
                # BGR (0 - 2^depth-1) -> RGB (0 - 1)
                rgb_vals = np.interp(np.flip(split_cols[i][0], -1),
                                     (0, main_script.max_val[split_cols[i][1][0][1]]), (0, 1))

                # RGB (0 - 1) -> XYZ D50 (0 - 1)
                xyz_vals = colour.RGB_to_XYZ(main_script.color_model[split_cols[i][1][0][0]]
                                             .cctf_decoding(rgb_vals),
                                             main_script.color_model[split_cols[i][1][0][0]].whitepoint,
                                             colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'],
                                             main_script.color_model[split_cols[i][1][0][0]].matrix_RGB_to_XYZ)
                # XYZ D50 (0 - 1) -> LAB D50 (0 - 100, -100 - 100, -100 - 100)
                lab_vals = colour.XYZ_to_Lab(xyz_vals, illuminant=colour.CCS_ILLUMINANTS[
                    'CIE 1931 2 Degree Standard Observer']['D50'])

                out_cols.append(lab_vals)

            elif conversion == 'adapt':  # LAB -> LAB(D50)
                # LAB (0 - 100, -100 - 100, -100 - 100) -> XYZ (0 - 1)
                xyz_vals = colour.Lab_to_XYZ(split_cols[i][0], illuminant=colour.CCS_ILLUMINANTS[
                    'CIE 1931 2 Degree Standard Observer'][in_cols[1]])

                # XYZ (0 - 1) -> XYZ D50 (0 - 1)
                xyz_vals = colour.chromatic_adaptation(xyz_vals, colour.xy_to_XYZ(
                    colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][in_cols[1]]),
                                                       colour.xy_to_XYZ(
                                                           colour.CCS_ILLUMINANTS[
                                                               'CIE 1931 2 Degree Standard Observer']['D50']))

                # XYZ D50 (0 - 1) -> LAB D50 (0 - 100, -100 - 100, -100 - 100)
                lab_vals = colour.XYZ_to_Lab(xyz_vals, illuminant=colour.CCS_ILLUMINANTS[
                    'CIE 1931 2 Degree Standard Observer']['D50'])

                out_cols.append(lab_vals)

            else:
                # LAB D50 (0 - 100, -100 - 100, -100 - 100) -> XYZ D50 (0 - 1)
                xyz_vals = colour.Lab_to_XYZ(split_cols[i][0], illuminant=colour.CCS_ILLUMINANTS[
                    'CIE 1931 2 Degree Standard Observer']['D50'])

                if conversion == 'out':  # LAB(D50) -> BGR(output)
                    # XYZ D50 (0 - 1) -> RGB (0 - 1)
                    rgb_vals = main_script.color_model[settings.output_color_space].cctf_encoding(colour.XYZ_to_RGB(
                        xyz_vals,
                        colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'],
                        main_script.color_model[settings.output_color_space].whitepoint,
                        main_script.color_model[settings.output_color_space].matrix_XYZ_to_RGB))

                    # RGB (0 - 1) -> BGR (0 - 2^depth-1)
                    bgr_vals = (np.interp(np.flip(rgb_vals, -1),
                                          (0, 1), (0, main_script.max_val[settings.output_depth]))
                                .astype(main_script.bit_type[settings.output_depth]))

                    out_cols.append(bgr_vals)

                elif conversion == 'show':  # LAB(D50) -> BGR(0-255 8bit sRGB)
                    # XYZ D50 (0 - 1) -> sRGB (0 - 1)
                    rgb_vals = main_script.color_model[0].cctf_encoding(colour.XYZ_to_RGB(
                        xyz_vals,
                        colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'],
                        main_script.color_model[0].whitepoint,
                        main_script.color_model[0].matrix_XYZ_to_RGB))

                    # sRGB (0 - 1) -> BGR (0 - 255)
                    bgr_vals = (np.interp(np.flip(rgb_vals, -1), (0, 1),
                                          (0, main_script.max_val[0]))
                                .astype(main_script.bit_type[0]))

                    out_cols.append(bgr_vals)

                elif conversion == 'LAB':  # LAB(D50) -> LAB(output)
                    # XYZ D50 (0 - 1) -> XYZ out (0 - 1)
                    xyz_vals = colour.chromatic_adaptation(xyz_vals, colour.xy_to_XYZ(
                        colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']),
                                                           colour.xy_to_XYZ(
                                                               colour.CCS_ILLUMINANTS[
                                                                   'CIE 1931 2 Degree Standard Observer'][
                                                                   settings.output_illuminant]))

                    # XYZ out (0 - 1) -> LAB out (0 - 100, -100 - 100, -100 - 100)
                    lab_vals = colour.XYZ_to_Lab(xyz_vals, illuminant=colour.CCS_ILLUMINANTS[
                        'CIE 1931 2 Degree Standard Observer'][settings.output_illuminant])

                    out_cols.append(lab_vals)

                elif conversion == 'RGB':  # LAB(D50) -> RGB(0.0-1.0 sRGB)
                    # XYZ D50 (0 - 1) -> sRGB (0 - 1)
                    rgb_vals = main_script.color_model[0].cctf_encoding(colour.XYZ_to_RGB(
                        xyz_vals,
                        colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'],
                        main_script.color_model[0].whitepoint,
                        main_script.color_model[0].matrix_XYZ_to_RGB))

                    rgb_vals = np.clip(rgb_vals, 0, 1)  # Clip to allowed range

                    out_cols.append(rgb_vals)

                elif conversion == 'xy':  # LAB(D50) -> xy
                    # XYZ D50 (0 - 1) -> xy (0 - 1)
                    xy_vals = colour.XYZ_to_xy(xyz_vals, illuminant=colour.CCS_ILLUMINANTS[
                        'CIE 1931 2 Degree Standard Observer']['D50'])

                    out_cols.append(xy_vals)

                else:
                    utilities.print_color(f"Invalid color conversion '{conversion}'!", 'error')

        out_cols = (np.concatenate(out_cols), in_cols[1])  # Combine split colors

        return out_cols
    else:
        # Perform multithreaded operations
        out_cols = image_utilities.parallel_process(convert_color, in_cols,
                                                    (conversion, True, settings.cpu_threads))
        return out_cols


# Correct image with LUT
def adjust_color(in_img, in_lut, is_thread=False, gray_diff=(0, 0, 0), operations=1):
    if is_thread or settings.cpu_threads == 0 or max(in_img[0].shape) < settings.cpu_threads:
        # Split image by memory usage
        if len(in_img[0].shape) > 1:
            memory_usage = np.prod(in_img[0].shape) * operations
            split_img = np.array_split(in_img[0], math.ceil(memory_usage / settings.system_memory))

            out_img = []
            for i in range(len(split_img)):
                split_img[i] = (split_img[i], in_img[1])  # Include metadata in split image

                # Convert image to flat format
                start_col = split_img[i][0]
                lab_i = np.array((np.take(start_col, 0, axis=2).flatten(),
                                  np.take(start_col, 1, axis=2).flatten(),
                                  np.take(start_col, 2, axis=2).flatten()))

                # Apply LUT and gray card difference
                fit_col = in_lut.apply(lab_i.T).reshape(start_col.shape[0], start_col.shape[1], 3) + gray_diff

                out_img.append(fit_col)

            out_img = (np.concatenate(out_img), in_img[1])  # Combine split image
        else:
            # Apply LUT and gray card difference
            fit_col = in_lut.apply(in_img[0]) + gray_diff
            out_img = (fit_col, in_img[1])

        return out_img
    else:
        # Perform multithreaded operation
        out_img = image_utilities.parallel_process(adjust_color, in_img,
                                                   (in_lut, True, gray_diff, settings.cpu_threads))
        return out_img


# Crop sample images
def crop_samples(sample_name, adjust=False, ref_gray=False):
    # Set sample path
    if len(sample_name.split('-')) > 1:
        if sample_name.split('-')[1].split('_')[0] == 'gray':
            ref_gray = True
        path = rf"Corrected Images\{sample_name.split('-')[0]}\{sample_name}"
    else:
        path = rf"Corrected Images\{sample_name}"
        for sub_path in os.listdir(os.path.join(settings.main_directory, path)):  # Crop each sample in directory
            if len(sub_path.split('.')) == 1:
                crop_samples(sub_path, adjust, ref_gray)
        return

    print()
    print("Cropping", sample_name)

    file_names = natsort.natsorted(os.listdir(os.path.join(settings.main_directory, path)))  # Get files in order

    if not ref_gray:  # If cropping reference gray, no need to handle reference data
        ref_crop_data = image_utilities.read_crop(file_names[0])
        if ref_crop_data is not None:
            start_file = ref_crop_data[1][0] + '.' + settings.output_extension
            # Check if cropped image exists, otherwise ignore existing data
            crop_exists = os.path.exists(os.path.join(rf'{os.path.join(settings.main_directory, path)}\Cropped',
                                                      start_file.split('.')[0] + '_cropped.'
                                                      + settings.output_extension))
        else:
            start_file = file_names[0]  # Use first file alphabetically
            crop_exists = False

        if not crop_exists or adjust:
            img = None
            while img is None:  # Wait for successful read
                img = image_utilities.read_image(start_file, os.path.join(settings.main_directory, path), convert=False)
            if adjust and ref_crop_data is not None and crop_exists:
                # Start with previous crop data
                ref_crop = match_crop(img, 0, (ref_crop_data[2], ref_crop_data[3]), convert=False)
            else:
                ref_crop = match_crop(img, 0, convert=False)
            if ref_crop is None:
                utilities.print_color("Discarding crop data.", 'warning')
                return

            if adjust and ref_crop_data is not None and crop_exists:
                start_points = match_crop(img, 1, ref_crop_data[1][1], convert=False)  # Start with previous crop data
            else:
                start_points = match_crop(img, 1, convert=False)
            if start_points is None:
                utilities.print_color("Discarding crop data.", 'warning')
                return
            img_scale = scale_image(img)[1]
        else:
            # Use existing data
            start_file = ref_crop_data[1][0] + '.' + settings.output_extension
            img = image_utilities.read_image(start_file, os.path.join(settings.main_directory, path), convert=False)
            img_scale = scale_image(img)[1]
            ref_crop = (ref_crop_data[2], ref_crop_data[3])
            start_points = ref_crop_data[1][1]

        ref_point_list = []
        gray_refs = None
    else:
        ref_point_list = []
        gray_refs = {}
        ref_crop_data = None
        img = None
        img_scale = None
        start_points = None
        start_file = None
        ref_crop = None

    img_refs = []
    write_dict = {}
    for i in range(len(file_names)):
        if len(file_names[i].split('.')) == 1:
            continue  # Skip folders

        # Check if crop data exists
        crop_exists = os.path.exists(os.path.join(rf'{os.path.join(settings.main_directory, path)}\Cropped',
                                                  file_names[i].split('.')[0] + '_cropped.'
                                                  + settings.output_extension))
        data_exists = False
        if ((ref_gray or (ref_crop_data is not None and file_names[i].split('.')[0] in ref_crop_data[0]))
                and crop_exists):
            data_exists = True
            if not adjust:
                continue

        if not ref_gray:
            if len(img_refs) == 0:
                # Show zoomed in reference points
                for o in range(2):
                    img_ref = zoom_image((np.interp(img[0],
                                                    (0, main_script.max_val[img[1][0][1]]), (0, main_script.max_val[0])
                                                    ).astype(main_script.bit_type[0]), img[1]),
                                         zoom_point=image_utilities.cvt_point(start_points[o], -1, img[0].shape))
                    cv.drawMarker(img_ref[0],
                                  image_utilities.cvt_point(start_points[o], -1, img_ref[0].shape),
                                  (0, 0, main_script.max_val[img_ref[1][0][1]]), cv.MARKER_TILTED_CROSS,
                                  round(15 / img_scale * 2), round(2 / img_scale * 2))  # Match regular size
                    img_refs.append(img_ref[0])

                image_utilities.show_image("Reference points", scale_image((np.concatenate(img_refs), img[1]))[0],
                                           False)

            img = image_utilities.read_image(file_names[i], os.path.join(settings.main_directory, path),
                                             convert=False)
            if file_names[i] != start_file:

                if data_exists:
                    ref_points = match_crop(img, 1, ref_crop_data[0][file_names[i].split('.')[0]], convert=False)
                else:
                    ref_points = match_crop(img, 1, convert=False)
            else:
                ref_points = start_points

            if ref_points is None:  # Esc pressed, skip sample
                print("Discarding crop data.")
                return

            ref_point_list.append((file_names[i].split('.')[0], ref_points))  # Save ref. points

            # Calculate angle and scale based on ref. points
            ref_angle = (utilities.get_angle(start_points[0], start_points[1])
                         - utilities.get_angle(ref_points[0], ref_points[1]))
            ref_scale = math.dist(start_points[0], start_points[1]) / math.dist(ref_points[0], ref_points[1])

            ref_translate = (0, 0)
            for o in range(2):
                # Convert to polar coordinates for applying calculated angle and scale
                ref_polar = image_utilities.cvt_point(ref_points[o], 2)
                new_ref = image_utilities.cvt_point((ref_polar[0] * ref_scale, ref_polar[1] + ref_angle), -2)
                ref_translate = np.sum((ref_translate, np.subtract(start_points[o], new_ref)), axis=0)

            ref_translate = np.divide(ref_translate, 2)  # Get average translation

            # Apply rotation, scaling, translation
            matched_img = translate_image(scale_image(rotate_image(img, ref_angle), ref_scale)[0],
                                          ref_translate)

            img_c = rotate_image(matched_img, ref_crop[0])  # Apply crop rotation
        else:  # No need to save cropping data for ref. grays
            if len(ref_point_list) == 0:
                ref_point_list.append([file_names[i].split('.')[0]])  # Save ref. gray names

            img = image_utilities.read_image(file_names[i], os.path.join(settings.main_directory, path), convert=False)

            ref_crop = match_crop(img, 0, convert=False)
            img_c = rotate_image(img, ref_crop[0])  # Apply crop rotation

        # Apply crop
        crop_corner = (image_utilities.cvt_point(ref_crop[1][0], -1, img_c[0].shape),
                       image_utilities.cvt_point(ref_crop[1][1], -1, img_c[0].shape))
        img_c = get_image_range(img_c, ((crop_corner[0][0], crop_corner[1][0]), (crop_corner[0][1], crop_corner[1][1])))

        write_dict[file_names[i]] = img_c  # Add image to writing dictionary

        if ref_gray:
            # Write average color
            gray_refs[file_names[i].split('.')[0].split('_')[1]] = tuple(elem for elem in
                                                                         image_utilities
                                                                         .get_average_color(convert_color(img_c, 'in')))

    if len(write_dict) > 0:
        image_utilities.write_crop(ref_point_list, ref_crop[0], ref_crop[1], gray_refs)
        print("Writing cropped images ...")
        for key in write_dict:
            # Write images in dictionary
            image_utilities.write_image(write_dict[key], key.split('.')[0], rf'{path}\Cropped',
                                        '_cropped.' + settings.output_extension, convert=False)
    else:
        print("No new data written.")

    cv.destroyAllWindows()


# Mode 0: Crop first image, 1: Match crop
def match_crop(in_img, mode=1, ref_points=(((0, 0), (0, 0)), ((0, 0), (0, 0))), convert=True):
    if convert:
        # Convert to sRGB 8bit
        in_img = convert_color(in_img, 'show')
    else:
        # Only convert to 8bit
        in_img = (np.interp(in_img[0], (0, main_script.max_val[in_img[1][0][1]]), (0, main_script.max_val[0])
                            ).astype(main_script.bit_type[0]), in_img[1])
    if mode == 0:  # Crop first image
        img_c, img_scale = scale_image(in_img)
        if ref_points[0] != ((0, 0), (0, 0)):
            # Previous rotation data exists -> Apply
            img_r = rotate_image(img_c, ref_points[0])
            ref_rotated = ref_points[0]
        else:
            # No rotation
            img_r = img_c
            ref_rotated = 0

        # Select horizontal line
        prompt = settings.prompts['horizontal']
        key_pressed = None
        ref_angle = None

        # Only accept rotation with none (= no rotation or previous rotation if adjusting) or both selection points
        while key_pressed is None or any(np.sum(elem) == -2 for elem in image_utilities.selected_points):
            image_utilities.show_image(prompt, img_r, False)
            cv.setMouseCallback(prompt, image_utilities.image_event, param=[prompt, img_r])
            key_pressed = image_utilities.wait_key()
            if key_pressed == 'escape':
                # Skip image
                return None
            elif key_pressed == 'space':
                # Use defaults
                break
            elif (ref_points[0] != ((0, 0), (0, 0)) and key_pressed == 'enter'
                  and all(np.sum(elem) == -2 for elem in image_utilities.selected_points)):
                # Use previous angle data
                ref_angle = ref_points[0]
                print(f"Using previous angle: {np.round(ref_angle, 2)} (deg)")
                break
        cv.destroyWindow(prompt)

        if key_pressed == 'space':
            # Default to no rotation
            print("Setting default rotation: 0 (deg).")
            ref_angle = 0
        else:
            if ref_angle is None:
                # Order selected points starting from left
                if image_utilities.selected_points[0][0] > image_utilities.selected_points[1][0]:
                    image_utilities.selected_points = (image_utilities.selected_points[1],
                                                       image_utilities.selected_points[0])

                ref_angle = utilities.get_angle(image_utilities.selected_points[0],
                                                image_utilities.selected_points[1]) + ref_rotated
            image_utilities.selected_points = ((-1, -1), (-1, -1))  # Clear selection

            img_c = rotate_image(img_c, ref_angle)  # Apply final rotation

        if ref_points[1] != ((0, 0), (0, 0)):
            # Get existing crop data
            roi = image_utilities.cvt_point(ref_points[1][0], -1, in_img[0].shape)
            roi = (roi[0] * img_scale, roi[1] * img_scale,
                   abs(ref_points[1][1][0] - ref_points[1][0][0]) * img_scale,
                   abs(ref_points[1][1][1] - ref_points[1][0][1]) * img_scale)

            # Show previous crop rectangle, select crop
            roi, key_pressed = image_utilities.get_roi(img_c, in_roi=roi, show_format=True)[1:3]
        else:
            roi, key_pressed = image_utilities.get_roi(img_c, show_format=True)[1:3]  # Select crop

        cv.destroyAllWindows()
        if key_pressed == 'escape':
            # Skip image
            return None

        if roi is not None:
            # Calculate crop corners
            ref_corners = np.divide((roi[0] - img_c[0].shape[1] / 2,
                                     img_c[0].shape[0] / 2 - roi[1],
                                     (roi[0] + roi[2]) - img_c[0].shape[1] / 2,
                                     img_c[0].shape[0] / 2 - (roi[1] + roi[3])), img_scale)
        else:
            if key_pressed != 'space' and ref_points[1] != ((0, 0), (0, 0)):
                # Use previous crop
                print("Using previous crop.")
                ref_corners = (ref_points[1][0][0], ref_points[1][0][1],
                               ref_points[1][1][0], ref_points[1][1][1])
            else:
                # Default crop to image corners (= no crop)
                print("Setting default crop: no crop.")
                ref_corners = np.divide((-img_c[0].shape[1] / 2,
                                         img_c[0].shape[0] / 2,
                                         img_c[0].shape[1] / 2,
                                         -img_c[0].shape[0] / 2), img_scale)

        # Round corner values
        ref_corners = ((round(ref_corners[0]), round(ref_corners[1])),
                       (round(ref_corners[2]), round(ref_corners[3])))

        return ref_angle, ref_corners

    elif mode == 1:  # Match crop
        img_c, img_scale = scale_image(in_img)  # Scale to max window size
        if np.sum(ref_points) != 0:
            # Scale existing ref. points
            image_utilities.selected_points = ([round(sel_point * img_scale) for sel_point in
                                                image_utilities.cvt_point(ref_points[0], -1,
                                                                          np.divide(img_c[0].shape, img_scale))],
                                               [round(sel_point * img_scale) for sel_point in
                                                image_utilities.cvt_point(ref_points[1], -1,
                                                                          np.divide(img_c[0].shape, img_scale))])
        # Select reference points
        prompt = settings.prompts['ref']
        key_pressed = None

        # Only accept reference with both selection points
        while key_pressed is None or any(np.sum(elem) == -2 for elem in image_utilities.selected_points):
            image_utilities.show_image(prompt, img_c, False)
            cv.setMouseCallback(prompt, image_utilities.image_event, param=[prompt, img_c])

            # Call image event to initialize window properly
            image_utilities.image_event(cv.EVENT_LBUTTONUP,
                                        image_utilities.selected_points[0][0], image_utilities.selected_points[0][1],
                                        None, [prompt, img_c])
            key_pressed = image_utilities.wait_key()
            if key_pressed == 'escape':
                # Skip image
                return None
            elif key_pressed == 'space':
                # Use defaults
                break
        cv.destroyWindow(prompt)

        if key_pressed == 'space':
            # Default reference to corners
            print("Setting default ref. points: corners.")
            ref_points = ((-in_img[0].shape[1] / 2, in_img[0].shape[0] / 2),
                          (in_img[0].shape[1] / 2, -in_img[0].shape[0] / 2))
        else:
            # Use selected reference points
            ref_points = np.divide((image_utilities.cvt_point(image_utilities.selected_points[0], 1, img_c[0].shape),
                                    image_utilities.cvt_point(image_utilities.selected_points[1], 1, img_c[0].shape)),
                                   img_scale)
            if ref_points[0][0] > ref_points[1][0]:
                ref_points = (ref_points[1], ref_points[0])

            image_utilities.selected_points = ((-1, -1), (-1, -1))  # Clear selection

        # Round ref. point values
        return [round(ref_point) for ref_point in ref_points[0]], [round(ref_point) for ref_point in ref_points[1]]
    else:
        print("Invalid cropping mode.")
        return None


# Crop calibration target with corner reference points
def crop_target(in_img, template, ref_grid):
    template = convert_color(template, 'show')  # Convert template to sRGB 8bit
    start_points = ((0, 0), (template[0].shape[1], template[0].shape[0]))
    img_refs = []
    for i in range(2):
        # Show reference selection
        img_ref = zoom_image(template, 4, start_points[i])
        img_refs.append(img_ref[0])

    image_utilities.show_image("Reference points", scale_image((np.concatenate(img_refs), template[1]))[0], False)
    ref_points = match_crop(in_img, 1)
    cv.destroyWindow("Reference points")
    angle = utilities.get_angle(ref_points[0], ref_points[1]) - ref_grid[2]
    new_ref = []
    for i in range(2):
        # Convert to corner coordinates
        ref_polar = image_utilities.cvt_point(ref_points[i], 2)
        new_ref.append(image_utilities.cvt_point(image_utilities.cvt_point((ref_polar[0], ref_polar[1] - angle), -2),
                                                 -1, in_img[0].shape))

    # Apply rotation and crop
    out_img = get_image_range(rotate_image(in_img, -angle), ((new_ref[0][0], new_ref[1][0]),
                                                             (new_ref[0][1], new_ref[1][1])))

    return out_img


# Offset image by given (x, y) values
def translate_image(in_img, offset):
    translate_matrix = np.float32([[1, 0, offset[0]], [0, 1, -offset[1]]])
    return cv.warpAffine(in_img[0].copy(), translate_matrix, (in_img[0].shape[1], in_img[0].shape[0])), in_img[1]


# Scale image by multiplier, or to max window size if left empty
def scale_image(in_img, size_multiplier=-1.0):
    img_s = (in_img[0].copy(), in_img[1])  # Don't edit original image
    if size_multiplier == -1.0:
        # Calculate maximum size multiplier
        img_scale = min(settings.max_window[0] / img_s[0].shape[1],
                        settings.max_window[1] / img_s[0].shape[0])
    else:
        img_scale = size_multiplier
    img_scaled = (cv.resize(img_s[0], None, fx=img_scale, fy=img_scale, interpolation=cv.INTER_AREA), img_s[1])

    return img_scaled, img_scale


# Zoom the image to specified point by given zooming factor
def zoom_image(in_img, zoom_factor=settings.selection_zoom, zoom_point=(-1, -1)):
    if zoom_point != (-1, -1):
        # Zoom to given point without showing areas outside image frame
        offset = np.divide((in_img[0].shape[1] / 2 - zoom_point[0], zoom_point[1] - in_img[0].shape[0] / 2),
                           1 + 1 / (zoom_factor - 1))
    else:
        # Zoom to center
        offset = np.array((0, 0))

    # Apply zoom
    out_img = scale_image((translate_image(in_img, offset)[0][
                           round(in_img[0].shape[0] / 2 - in_img[0].shape[0] / zoom_factor / 2):
                           round(in_img[0].shape[0] / 2 + in_img[0].shape[0] / zoom_factor / 2),
                           round(in_img[0].shape[1] / 2 - in_img[0].shape[1] / zoom_factor / 2):
                           round(in_img[0].shape[1] / 2 + in_img[0].shape[1] / zoom_factor / 2)],
                           in_img[1]), zoom_factor)[0]
    return out_img


# Draw transparent arrow on image
def draw_arrow(in_img, coords, arrow_width):
    line_layer = in_img[0].copy()
    cv.arrowedLine(line_layer, coords[0], coords[1],
                   (0, 0, main_script.max_val[in_img[1][0][1]] / 2), arrow_width,
                   tipLength=(25 / (math.dist(coords[0], coords[1]) + 1)))
    return (cv.addWeighted(line_layer, settings.arrow_alpha, in_img[0], 1 - settings.arrow_alpha, 0), in_img[1])


# Rotate image by given angle (in mathematically positive direction = counter-clockwise)
def rotate_image(in_img, rot_angle):
    image_center = tuple(np.array(in_img[0].shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, rot_angle, 1.0)

    # Apply rotation to copy of given image
    img_rot = (cv.warpAffine(in_img[0].copy(), rot_mat, in_img[0].shape[1::-1], flags=cv.INTER_LINEAR), in_img[1])

    return img_rot


# Get given range of image
def get_image_range(in_img, in_range):  # in_range format (corner): ((left x, right x), (top y, bottom y))
    in_range = ([round(r_pos) for r_pos in in_range[0]], [round(r_pos) for r_pos in in_range[1]])

    # Calculate amount of overflow
    oversize = ((max(-in_range[0][0], 0), max(in_range[0][1] - in_img[0].shape[1], 0)),
                (max(-in_range[1][0], 0), max(in_range[1][1] - in_img[0].shape[0], 0)))

    out_img = (in_img[0][max(0, in_range[1][0]):min(in_img[0].shape[0], in_range[1][1]),
               max(0, in_range[0][0]):min(in_img[0].shape[1], in_range[0][1])], in_img[1])

    # Set pixel values outside image frame to black
    if oversize[0][0] > 0:
        out_img = (np.concatenate((np.zeros((out_img[0].shape[0], oversize[0][0], 3),
                                            dtype=main_script.bit_type[settings.output_depth]), out_img[0]), axis=1),
                   out_img[1])
    if oversize[0][1] > 0:
        out_img = (np.concatenate((out_img[0], np.zeros((out_img[0].shape[0], oversize[0][1], 3),
                                                        dtype=main_script.bit_type[settings.output_depth])), axis=1),
                   out_img[1])
    if oversize[1][0] > 0:
        out_img = (np.concatenate((np.zeros((oversize[1][0], out_img[0].shape[1], 3),
                                            dtype=main_script.bit_type[settings.output_depth]), out_img[0]), axis=0),
                   out_img[1])
    if oversize[1][1] > 0:
        out_img = (np.concatenate((out_img[0], np.zeros((oversize[1][1], out_img[0].shape[1], 3),
                                                        dtype=main_script.bit_type[settings.output_depth])), axis=0),
                   out_img[1])

    return out_img
