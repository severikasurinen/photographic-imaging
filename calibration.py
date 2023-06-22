import settings
import utilities
import image_utilities
import main_script
import image_manipulation

import time
import numpy as np
import cv2 as cv
import colour
from matplotlib import pyplot as plt
plt.rcParams["figure.dpi"] = 200


def color_calibration(in_img, ref_data, show_img=False):
    """Create 3D LUT from input image and reference data"""
    rec = image_manipulation.convert_color(in_img, 'show')

    # Base for color plot
    colour.plotting.plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931(
        main_script.color_model[settings.output_color_space], standalone=False)

    # Calculate sample average colors
    fit_data = np.zeros(6)
    sample_data = np.zeros((4, ref_data[0][0][0], ref_data[0][0][1]))
    lab_vals = [[[], []], [[], []], [[], []]]
    lab_range = [[100, 100, 100], [0, -100, -100]]
    labels = ('L*', 'a*', 'b*', 'CIELAB')
    for l_y in range(ref_data[0][0][1]):
        for l_x in range(ref_data[0][0][0]):
            # Find top left corner of sample
            top_left = (round((ref_data[0][1][0][0] + ref_data[0][1][1][0] * l_x)
                              * (in_img[0].shape[1] / ref_data[0][0][0])),
                        round((ref_data[0][1][0][1] + ref_data[0][1][1][1] * l_y)
                              * (in_img[0].shape[0] / ref_data[0][0][1])))

            # Find bottom right corner of sample
            bottom_right = (round(top_left[0] + ref_data[0][1][2][0] * (in_img[0].shape[1] / ref_data[0][0][0])),
                            round(top_left[1] + ref_data[0][1][2][1] * (in_img[0].shape[0] / ref_data[0][0][1])))
            rec = (cv.rectangle(rec[0], top_left, bottom_right, (0, 0, 255), 2), rec[1])        # Draw rectangle
            grid_sample = in_img[0][top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]   # Crop sample
            avg_col = image_utilities.get_average_color((grid_sample, in_img[1]))               # Sample average color
            sample_data[0, l_x, l_y] = colour.delta_E(avg_col, ref_data[1][l_x][l_y], method='CIE 2000')
            print(f'({l_x + 1}, {l_y + 1})', f"Average color - LAB ({settings.output_illuminant}):",
                  image_manipulation.convert_color((avg_col, in_img[1]), 'LAB')[0].round(3))
            print("Reference CIEDE2000:", "{:.3f}".format(sample_data[0, l_x, l_y]))

            ref_xy = image_manipulation.convert_color((ref_data[1][l_x][l_y], in_img[1]), 'xy')[0]  # Reference xy
            avg_xy = image_manipulation.convert_color((avg_col, in_img[1]), 'xy')[0]                # Sample xy

            # Plot line from reference point
            plt.plot((ref_xy[0], avg_xy[0]), (ref_xy[1], avg_xy[1]), 'k')
            plt.plot(ref_xy[0], ref_xy[1], '.w')

            # Plot separate points
            # plt.plot(ref_xy[0], ref_xy[1], '.k')
            # plt.plot(avg_xy[0], avg_xy[1], '.w')

            for i in range(3):
                # Append sample color
                lab_vals[i][0].append(avg_col[i])
                lab_vals[i][1].append(ref_data[1][l_x][l_y][i])

                # Sample deviation from reference data
                sample_data[i + 1, l_x, l_y] = np.subtract(lab_vals[i][0][-1], lab_vals[i][1][-1])

                # Save LAB ranges
                lab_range[0][i] = min(lab_range[0][i], lab_vals[i][0][-1])
                lab_range[1][i] = max(lab_range[1][i], lab_vals[i][0][-1])

    for i in range(3):
        # Calculate average differences
        fit_data[i + 3] = np.average(np.average(np.abs(sample_data[i + 1])))

        # Check if LAB are out of domain (without considering color ranges for specific luminance)
        if lab_range[0][i] < settings.input_lab_domain[0][i] or lab_range[1][i] > settings.input_lab_domain[1][i]:
            utilities.print_color("Input " + labels[i] + " out of domain: (" + str(lab_range[0][i]) + ', ' +
                                  str(lab_range[1][i]) + ')!', 'error')

    fit_data[0] = np.average(sample_data[0])            # Average CIEDE2000
    fit_data[1] = sample_data[0].min(initial=1000.0)    # Min. CIEDE2000
    fit_data[2] = sample_data[0].max(initial=0.0)       # Max. CIEDE2000

    print()
    print("Avg. luminance difference:", fit_data[3].round(3))
    print("Avg. a difference:", fit_data[4].round(3))
    print("Avg. b difference:", fit_data[5].round(3))

    print("Ref. average CIEDE2000:", "{:.3f}".format(fit_data[0]))
    print("Ref. CIEDE2000 range: [", "{:.3f}".format(fit_data[1]), "-",
          "{:.3f}".format(fit_data[2]), "]")

    # Show color plot
    plt.title(f'CIEDE2000: {fit_data[0].round(3)} [{fit_data[2].round(3)}]')
    plt.show()

    correction_lut = None

    if show_img:
        # Show corrected color target
        image_utilities.show_image("Corrected target (Press Esc to discard, or any other key to save)",
                                   image_manipulation.scale_image(rec)[0], False)
    else:
        s_time = time.perf_counter()

        # Create 3D LUT
        correction_lut = utilities.create_lut((lab_vals[0][0], lab_vals[1][0], lab_vals[2][0]),
                                              (lab_vals[0][1], lab_vals[1][1], lab_vals[2][1]),
                                              settings.input_lab_domain)

        print('LUT creation time:', time.perf_counter() - s_time, '(s)')

        # Plot calibration result
        plot_domain = ((0, -60, -60), (100, 60, 60))    # Plot input range
        edges = []
        step_size = 2   # Plot step size

        # Create grid for plotting color values
        for o in range(3):
            edges.append(np.linspace(plot_domain[0][o], plot_domain[1][o],
                                     round((plot_domain[1][o] - plot_domain[0][o]) / step_size)))

        lab_i = np.array(np.meshgrid(edges[0], edges[1], edges[2]))
        lab_i = np.array((lab_i[0].flatten(), lab_i[1].flatten(), lab_i[2].flatten()))
        res_i = correction_lut.apply(lab_i.T)   # Calculate color values at grid points

        # Plot 3D LUT cubes
        for i in range(4):
            fig = plt.figure()
            plt.title(labels[i])
            ax = fig.add_subplot(111, projection='3d')
            if i < 3:
                # Plot single value cubes
                fig.colorbar(ax.scatter(lab_i[1], lab_i[2], lab_i[0], c=res_i.take(i, axis=1), cmap=plt.viridis()),
                             shrink=0.9, pad=0.1)
            else:
                # Plot cube with mapped colors
                ax.scatter(lab_i[1], lab_i[2], lab_i[0],
                           c=image_manipulation.convert_color((res_i, in_img[1]), 'RGB')[0])
            l, b, w, h = ax.get_position().bounds
            ax.set_position([l, b + 0.05 * h, w, h * 0.9])
            ax.set_xlabel('a*')
            ax.set_ylabel('b*')
            ax.set_zlabel('L*')
            ax.zaxis.labelpad = 0
            plt.show()

    print()
    return correction_lut, fit_data, sample_data


def image_uniformity(in_img):
    """Calculate image color uniformity"""
    image_utilities.show_image("Selected area", image_manipulation.scale_image(in_img)[0])

    # Calculate average color
    avg_col = image_utilities.get_average_color(in_img)
    print(f"Average color - LAB ({settings.output_illuminant}):",
          image_manipulation.convert_color((avg_col, in_img[1]), 'LAB')[0].round(3))

    # Downscale image for calculations
    img_u = image_manipulation.scale_image(in_img, settings.uniformity_scale)[0]
    # Calculate CIEDE2000 values for scaled image
    deltas = colour.delta_E(avg_col, img_u[0], method='CIE 2000')

    print("Average CIEDE2000:", "{:.3f}".format(np.average(deltas)))
    print("CIEDE2000 range: [", "{:.3f}".format(deltas.min(initial=1000.0)), "-",
          "{:.3f}".format(deltas.max(initial=0.0)), "]")

    px_range = [0, 0]
    img_d = np.zeros((img_u[0].shape[0], img_u[0].shape[1], 3))
    for y in range(img_u[0].shape[0]):
        for x in range(img_u[0].shape[1]):
            # Map CIEDE2000 to pixel value
            px_brightness = round(np.interp(deltas[y, x], (0, settings.ciede_max),
                                            (0, main_script.max_val[0])))
            if img_u[0][y, x][0] > avg_col[0]:
                img_d[y, x] = (0, 0, px_brightness)  # Red: Brighter
                px_range[1] = max((px_range[1], px_brightness))
            else:
                img_d[y, x] = (px_brightness, 0, 0)  # Blue: Darker
                px_range[0] = min((px_range[0], -px_brightness))
    img_d = img_d.astype(main_script.bit_type[0])   # Save as 8bit
    print("Image range:", px_range)

    return img_d, ((0, 0), {})
