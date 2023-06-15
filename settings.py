import multiprocessing


max_window = [1400, 800]        # width, height (px), window size limits
selection_zoom = 8              # Zoom ratio for selecting sample reference points
arrow_alpha = 0.5               # Transparency of measurement arrow
use_colored_printing = False    # Set to True if terminal used supports colored printing

prompt_focus_height = False     # Ask user to input focus height? (Not in use, requires new interpolation algorithm)
check_capture_settings = False  # Check if images match calibration capture settings?
save_gray_images = False        # Keep ref. gray images?
gray_warning_limit = 1.5        # ref. gray min. dE for warning

input_extension = 'tif'         # Input file extension

output_illuminant = 'D50'       # 'D50' (recommended, common in printing), 'D65' (common in other applications)
output_color_space = 2          # 0: sRGB, 1: Adobe RGB, 2: ProPhoto RGB (recommended)
output_depth = 1                # 0: 8bit - 1: 16bit (recommended)
output_extension = 'tif'        # Output file extension

uniformity_scale = 0.02         # Uniformity image resolution
ciede_max = 2.55                # CIEDE2000 value interpolated to max brightness

# Range of input LAB, max. range: (0, -100, -100), (100, 100, 100), natural colors usually (0, -40, -50), (100, 50, 60)
input_lab_domain = ((0, -100, -100), (100, 100, 100))
correction_epsilon = 1.0                            # Epsilon: smooth -> sharp curves
lut_size = 128                                      # Create 3D LUT of dimensions (lut_size x lut_size x lut_size x 3)

prompt_margin_utilization = False                   # Ask user if safety margins should be used? If not, will be used.
ref_margins = (0.15, 0.1)                           # Safety margins for reference gray images (x, y)

color_spaces = ('sRGB', 'Adobe', 'ProPhoto')    # Supported color spaces
bit_depths = (8, 16)                                # Supported bit depths
reference_types = ('colorchecker', 'it87')              # Supported reference types

# EXIF data to save
exif_data = ('EXIF:XResolution', 'EXIF:YResolution', 'EXIF:ResolutionUnit', 'EXIF:Make', 'EXIF:Model', 'EXIF:FNumber',
             'EXIF:ExposureTime', 'EXIF:ISO', 'EXIF:ExposureCompensation', 'EXIF:FocalLength', 'EXIF:MaxApertureValue',
             'EXIF:MeteringMode', 'EXIF:Flash', 'EXIF:FocalLengthIn35mmFormat', 'EXIF:Contrast', 'EXIF:BrightnessValue',
             'EXIF:ExposureMode', 'EXIF:Saturation', 'EXIF:Sharpness', 'EXIF:WhiteBalance', 'EXIF:DigitalZoomRatio',
             'EXIF:ExifVersion', 'EXIF:DateTimeOriginal')

main_directory = r'..\Images'                   # Main directory for images
directories = (r'Calibration\Image Uniformity', r'Calibration\Correction Profiles', r'Calibration\Spectra',
               r'Calibration\Calibration Images', r'Calibration\Reference Values', 'Corrected Images',
               'Exported Images', 'Remote Capture', 'Imaging Edge')

system_memory = 16000000                        # Bytes of system memory to utilize
cpu_threads = multiprocessing.cpu_count() - 1   # Number of threads to utilize, 0 for no parallel processing

# Window prompts
prompts = {'horizontal': "Drag horizontal line, then press ENTER.",
           'crop': "Select crop, then press ENTER.",
           'ref': "Select reference points, then press ENTER.",
           'ref-timelapse': "Select reference points, then press ENTER. "
                            "To use the same ref. points for rest of images, press A prior to ENTER.",
           'line': "Drag line to measure, then press ENTER. "
                   "You can adjust the width of the measured line using the mouse wheel."}

# Console print color list
print_colors = {'error': '\033[91m', 'green': '\033[92m', 'warning': '\033[93m', 'blue': '\033[94m', 'end': '\033[0m'}
