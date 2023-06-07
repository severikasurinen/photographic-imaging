import settings
import utilities
import image_utilities
import image_manipulation
import calibration

import os
import time
import numpy as np
import cv2 as cv  # Import OpenCV library
import colour  # Import Colour Science library
import natsort

max_val = (pow(2, 8) - 1, pow(2, 16) - 1)  # 8bit: 0-255 - 16bit: 0-65635
# Color models corresponding to color spaces in settings.py
color_model = (colour.models.RGB_COLOURSPACE_sRGB, colour.models.RGB_COLOURSPACE_ADOBE_RGB1998,
               colour.models.RGB_COLOURSPACE_PROPHOTO_RGB, colour.models.RGB_COLOURSPACE_ACES2065_1)
bit_type = ('uint8', 'uint16')


def main():
    print()
    utilities.create_directories(settings.main_directory)   # Create imaging system directory structure
    start_time = 0
    end_time = 0

    # Start in main menu
    while True:
        script_mode = input("Script mode (0: Apply corrections, 1: Calibrate, 2: Crop images, 3: Measure series color"
                            ", ENTER: Exit): ")
        if script_mode is not None:
            script_mode = script_mode.strip()
        if script_mode == '':
            print("Closing program.")
            return

        script_mode = int(script_mode)
        if 0 <= script_mode <= 3:
            # Invalid input
            break
        else:
            print("Invalid mode.")

    if script_mode == 0:  # Apply corrections
        while True:
            if settings.prompt_focus_height:
                # Ask user for focus height
                focus_input = input("Height - focus height (mm): ")
                if focus_input is not None:
                    focus_input = focus_input.strip()
            else:
                # Use default height
                focus_input = ''

            profile_data = image_utilities.read_profile(focus_input)
            if profile_data is not None:
                break
            else:
                print("Invalid correction profile, only use numbers.")
        print()

        start_time = time.perf_counter()

        # Sort files alphabetically
        file_names = natsort.natsorted(os.listdir(os.path.join(settings.main_directory, 'Exported Images')))
        gray_files = []
        study_names = []
        print("Correcting gray references ...")
        print()
        use_margins = None
        for file_name in file_names:
            if len(file_name.split('-')) > 1 and file_name.split('-')[1].split('_')[0] == "gray":
                # File name corresponds to ref. gray
                gray_files.append(file_name)
                if study_names.count(file_name.split('_')[0]) < 1:
                    study_names.append(file_name.split('_')[0])
                if use_margins is None:
                    use_margins = utilities.yes_no_prompt("Use safety margins?")
                if use_margins:
                    # Get area in safety margins
                    img_gray = image_utilities.get_safe_area(image_utilities.read_image(file_name))
                else:
                    # Use full image
                    img_gray = image_utilities.read_image(file_name)

                # Correct ref. gray, write result
                image_utilities.write_image(image_manipulation.adjust_color(img_gray, profile_data[0]),
                                            file_name.split('.')[0], image_utilities.sample_path(file_name),
                                            '_cc.' + settings.output_extension)
                os.remove(os.path.join(settings.main_directory, 'Exported Images', file_name))  # Remove original

        # Crop ref. grays
        for study_name in study_names:
            image_manipulation.crop_samples(study_name, ref_gray=True)

        file_names = natsort.natsorted(os.listdir(os.path.join(settings.main_directory, 'Exported Images')))
        sample_names = []
        est_data = [time.perf_counter(), 0]
        for i in range(len(file_names)):
            img = image_utilities.read_image(file_names[i])

            if settings.check_capture_settings and not image_utilities.compare_settings(img[1][1], profile_data[2]):
                print("Skipping", file_names[i], "(Capture settings mismatch)")
                continue

            print("Correcting", file_names[i], '...')

            path = image_utilities.sample_path(file_names[i])
            if not os.path.exists(os.path.join(settings.main_directory, path)):
                os.makedirs(os.path.join(settings.main_directory, path))

            measurement_gray = image_utilities.read_crop(file_names[i].split('-')[0] + "-gray", True)
            if measurement_gray is None:
                ref_diff = (0, 0, 0)
                utilities.print_color("Warning: Ref. gray not found!", 'warning')
                print()
            else:
                # Read ref. gray values
                if int(file_names[i].split('.')[0].split('_')[1]) in measurement_gray:
                    measurement_gray = measurement_gray[int(file_names[i].split('.')[0].split('_')[1])]
                else:
                    # Read the earliest ref. gray values after measurement or latest ref. gray values if not numbered
                    latest_gray = list(measurement_gray.keys())[-1]
                    for j in range(len(measurement_gray)):
                        latest_gray = list(measurement_gray.keys())[j]
                        try:
                            if int(list(measurement_gray.keys())[j]) > int(file_names[i].split('.')[0].split('_')[1]):
                                break
                        except ValueError:
                            latest_gray = list(measurement_gray.keys())[-1]
                            break

                    measurement_gray = measurement_gray[latest_gray]
                    utilities.print_color("Warning: Ref. gray not found!" + " Using ref. gray from measurement '"
                                          + str(latest_gray) + "'.", 'warning')
                    print()
                ref_diff = np.subtract(profile_data[1], measurement_gray)
                ref_ciede = colour.delta_E(profile_data[1], measurement_gray, method='CIE 2000').round(2)
                if ref_ciede >= settings.gray_warning_limit:
                    utilities.print_color(f"Warning: Ref. gray dE is {ref_ciede}!", 'warning')
                else:
                    utilities.print_color(f"Ref. gray dE OK, {ref_ciede}.", 'green')

            # Write corrected image
            image_utilities.write_image(image_manipulation.adjust_color(img, profile_data[0], gray_diff=ref_diff),
                                        file_names[i].split('.')[0], path, '_cc.' + settings.output_extension)
            os.remove(os.path.join(settings.main_directory, 'Exported Images', file_names[i]))  # Remove original

            if sample_names.count(file_names[i].split('_')[0]) == 0:
                sample_names.append(file_names[i].split('_')[0])

            est_data[1] += 1
            if est_data[1] == 1:
                # Estimate total processing time based on first image
                utilities.print_estimate(est_data[0], est_data[1] / len(file_names))

        if len(sample_names) > 0:
            print()
            if utilities.yes_no_prompt("Crop corrected images?"):
                for sample_name in sample_names:
                    image_manipulation.crop_samples(sample_name)

        end_time = time.perf_counter()

    elif script_mode == 1:  # Calibration
        while True:
            calib_mode = input("Measuring mode (0: Create calibration profile, 1: Measure image uniformity"
                               ", ENTER: Main menu): ")
            if calib_mode is not None:
                calib_mode = calib_mode.strip()
            if calib_mode == '':
                main()  # Return to main menu
                return

            calib_mode = int(calib_mode)
            if 0 <= calib_mode <= 1:
                break
            else:
                print("Invalid mode.")

        if calib_mode == 0:     # Create calibration profile
            while True:
                # Prompt for color target type
                ref_name = settings.reference_types[0]
                ref_data = None
                if len(settings.reference_types) > 1:
                    ref_name = input("Reference type: ")
                    if ref_name.strip() == '':
                        # Set default target
                        ref_name = 'it87'
                    ref_data = image_utilities.read_reference(ref_name)
                if ref_data is not None:
                    break
                else:
                    print("Invalid reference.")

            while True:
                # Prompt for focus height of calibration image
                focus_height = input("Height - focus height (mm): ")
                if focus_height is not None:
                    focus_height = focus_height.strip()
                img = image_utilities.read_image('calib_' + focus_height + '.' + settings.input_extension,
                                                 r'Calibration\Calibration Images')
                if img is not None:
                    break
                else:
                    print("Calibration image not found.")

            start_time = time.perf_counter()

            # Get reference gray image
            gray_img = image_utilities.read_image('calib-gray_' + focus_height + '.' + settings.input_extension,
                                                  r'Calibration\Calibration Images')

            if gray_img is None:
                print()
                utilities.print_color("Warning: Ref. gray not found!", 'warning')
                print()
            else:
                if utilities.yes_no_prompt("Use safety margins?"):
                    gray_img = image_utilities.get_safe_area(gray_img)  # Crop to safety margins
                ref_crop = image_manipulation.match_crop(gray_img, 0)   # Prompt for crop
                lt_corner = image_utilities.cvt_point(ref_crop[1][0], -1, gray_img[0].shape)    # Upper left corner

                # Rotate and crop as selected
                gray_img = image_utilities.get_roi(image_manipulation.rotate_image(gray_img, ref_crop[0]), 1,
                                                   in_roi=(lt_corner[0], lt_corner[1],
                                                           abs(ref_crop[1][1][0] - ref_crop[1][0][0]),
                                                           abs(ref_crop[1][1][1] - ref_crop[1][0][1])))[0]

            image_utilities.show_image("Loaded image", image_manipulation.scale_image(img)[0])  # Show original

            # Crop color target
            target_c = image_manipulation.crop_target(img, image_utilities.read_image(
                ref_name + '.jpg', r'Calibration\Reference Values'), ref_data[0][0])

            correction_lut = calibration.color_calibration(target_c, ref_data)[0]   # Create 3D LUT

            print()
            target_a = image_manipulation.adjust_color(target_c, correction_lut)    # Correct color target

            # Show corrected color target and get accuracy
            fit_data, sample_data = calibration.color_calibration(target_a, ref_data, True)[1:3]
            print()

            # Get ref. gray LAB values
            gray_lab = (0, 0, 0)
            if gray_img is not None:
                gray_avg = image_manipulation.adjust_color((image_utilities.get_average_color(gray_img), gray_img[1]),
                                                           correction_lut)
                gray_lab = tuple(elem for elem in gray_avg[0])

                print(f"Ref. gray LAB ({settings.output_illuminant}):"
                      f"{image_manipulation.convert_color(gray_avg, 'LAB')}")
                print()

            print("Correction LUT:", correction_lut)

            end_time = time.perf_counter()
            key_pressed = image_utilities.wait_key()
            cv.destroyAllWindows()

            if key_pressed != 'escape':
                # Save 3D LUT
                image_utilities.write_profile(focus_height, correction_lut, gray_lab, img[1],
                                              fit_data, sample_data, ref_data[0][0])
            else:
                print("Discarding profile data.")

        elif calib_mode == 1:  # Measure image uniformity
            use_margins = utilities.yes_no_prompt("Use safety margins?")

            while True:
                # Prompt for file to measure
                file_name = input("Image file name: ")

                # Get path of file
                path = image_utilities.sample_path(file_name)
                if os.path.exists(os.path.join(settings.main_directory, path)):
                    for file in os.listdir(os.path.join(settings.main_directory, path)):
                        file = str(file)

                        if file_name == file.split('.')[0]:
                            file_name = file
                            break
                        elif len(file_name.split('_')) <= len(file.split('_')):
                            is_same = True
                            for i in range(len(file_name.split('_'))):
                                if file_name.split('_')[i] != file.split('_')[i]:
                                    is_same = False

                            if is_same:
                                file_name = file
                                break

                    img = image_utilities.read_image(file_name, path)
                    if img is not None:
                        break

                print("Invalid image.")

            print()

            start_time = time.perf_counter()

            if use_margins:
                img = image_utilities.get_safe_area(img)    # Crop to safety margins
            img_uniformity = calibration.image_uniformity(image_utilities.get_roi(img)[0])

            image_utilities.show_image("Image uniformity", image_manipulation.scale_image(img_uniformity)[0],
                                       convert=False)

            end_time = time.perf_counter()
            key_pressed = image_utilities.wait_key()
            cv.destroyAllWindows()

            if key_pressed != 'escape':
                # Save uniformity image after closing window
                image_utilities.write_image(img_uniformity, file_name.split('.')[0], r'Calibration\Image Uniformity',
                                            '_uniformity.' + settings.output_extension, True, convert=False)
                print("Uniformity image saved.")
            else:
                print("Discarding uniformity image.")

    elif script_mode == 2:  # Crop images
        while True:
            # Prompt for sample name
            sample_name = input("Sample name: ")
            if len(sample_name.split('_')[0].split('-')) > 1:
                path = rf"Corrected Images\{sample_name.split('_')[0].split('-')[0]}\{sample_name}"
            else:
                path = rf"Corrected Images\{sample_name}"
            if os.path.exists(os.path.join(settings.main_directory, path)):
                break
            else:
                print("Invalid sample.")

        adjusting = utilities.yes_no_prompt("Adjust crop?")     # If not, images with existing crop data will be skipped

        start_time = time.perf_counter()

        image_manipulation.crop_samples(sample_name, adjusting)     # Crop sample image(s)

        end_time = time.perf_counter()

    elif script_mode == 3:  # Measure series color
        while True:
            # Prompt for measurement type
            measure_mode = input("Measuring mode (0: Measure area, 1: Measure along line, ENTER: Main menu): ")
            if measure_mode is not None:
                measure_mode = measure_mode.strip()
            if measure_mode == '':
                # Return to main menu
                main()
                return

            measure_mode = int(measure_mode)
            if 0 <= measure_mode <= 1:
                break
            else:
                print("Invalid mode.")

        while True:
            # Prompt for sample name
            img_name = input("Sample name: ")
            if len(img_name.split('_')[0].split('-')) > 1:
                path = rf"Corrected Images\{img_name.split('_')[0].split('-')[0]}\{img_name}"
            else:
                path = rf"Corrected Images\{img_name.split('_')[0]}"
            if img_name.strip() != '' and os.path.exists(os.path.join(settings.main_directory, path + r"\Cropped")):
                path = path + r"\Cropped"
                break
            elif img_name.strip() != '' and os.path.exists(os.path.join(settings.main_directory, path)):
                break
            else:
                print("Invalid sample.")

        while True:
            # Prompt for image to use as dE reference
            ref_name = input("Reference image name: ")
            if ref_name == '':
                for file_name in natsort.natsorted(os.listdir(os.path.join(settings.main_directory, path))):
                    if len(file_name.split('.')) > 1 and file_name.split('.')[1] == settings.output_extension:
                        ref_name = file_name.split('.')[0]
                        print("Reference image:", ref_name)
                        break

            if os.path.exists(os.path.join(settings.main_directory, path, ref_name + '.' + settings.output_extension)):
                break
            else:
                print("Invalid image.")

        while True:
            # Prompt for name of measurement
            measurement_name = input("Color measurement name: ")
            if measurement_name.strip() != '' and measurement_name == measurement_name.strip():
                break
            else:
                print("Invalid name. Don't include spaces.")

        start_time = time.perf_counter()

        image_utilities.measure_series(path, ref_name, measure_mode, measurement_name)  # Run measurement process

        end_time = time.perf_counter()

    else:
        print("Invalid script mode.")
        return

    print()
    print(f"Finished in {round(end_time - start_time, 2)} s.")

    main()  # Restart program at main menu


if __name__ == '__main__':
    main()
