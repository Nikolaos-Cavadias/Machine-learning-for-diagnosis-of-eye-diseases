from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, square, dilation
import os
import seaborn as sns
from fitter import Fitter, get_common_distributions
from scipy.stats import norm, lognorm, beta, burr, gamma

# Define the path to the local folder
folder_path = r"/mnt/iusers01/fse-ugpgt01/mace01/x93189nc/eyes/test + train/"

# List all files in the directory
file_list = os.listdir(folder_path)

# Define output directory for plots
output_dir = r"/mnt/iusers01/fse-ugpgt01/mace01/x93189nc/eyes/Plots_D/"
os.makedirs(output_dir, exist_ok=True)

# Define output directory for parameter files
param_output_dir = os.path.join(output_dir, "Parameters")
os.makedirs(param_output_dir, exist_ok=True)

# Filter the list to include only image files (assuming common image file extensions)
image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
image_files = [file for file in file_list if file.lower().endswith(image_extensions)]

# Define categories
category_A = []
category_N = []
category_G = []
category_D = []

# Function to get the character before the file extension
def get_category_character(filename):
    base_name = os.path.splitext(filename)[0]  # Remove the extension
    last_segment = base_name.split('_')[-1]  # Get the last segment after splitting by '_'
    return last_segment[0]  # Return the first character of the last segment

# Categorize the images based on the character before the file extension
for image_file in image_files:
    category_char = get_category_character(image_file)
    if category_char == 'A':
        category_A.append(image_file)
    elif category_char == 'N':
        category_N.append(image_file)
    elif category_char == 'G':
        category_G.append(image_file)
    elif category_char == 'D':
        category_D.append(image_file)

# Print categorized image lists and their sizes
print("Size of Category A:", len(category_A))
print("Size of Category N:", len(category_N))
print("Size of Category G:", len(category_G))
print("Size of Category D:", len(category_D))




# Function to save parameters to text files
def save_parameters_to_file(image_params_dict, category):
    param_file_path = os.path.join(param_output_dir, f"{category}_parameters.txt")
    with open(param_file_path, 'w') as file:
        for image_file, params in image_params_dict.items():
            file.write(f"{image_file}\n")
            for param, value in params.items():
                file.write(f"{param}: {value}\n")
            file.write("\n")




# Process only the first 5 images from each category
def process_images(image_files, category_name):
    image_params_dict = {}

#   for image_file in image_files[:5]       if u want the first 5 images
    for image_file in image_files :
        image_path = os.path.join(folder_path, image_file)
        segmentation = Image.open(image_path).convert('L')
        segmentation = np.array(segmentation) / 255  # Convert to numpy array with values 0 and 1

        # Compute the skeleton of the segmentation
        skeleton = skeletonize(segmentation > 0)

        # Assuming single-channel image for skeleton and segmentation
        binary_array = (skeleton > 0).astype(np.uint8)
        segmentation2d = segmentation

        # Instantiate geometrical VBM object
        from PVBM.GeometricalAnalysis import GeometricalVBMs
        geometricalVBMs = GeometricalVBMs()
        a = geometricalVBMs.area(segmentation2d)

        n_end, n_inter, end, inter = geometricalVBMs.compute_particular_points(binary_array)

        # Dilate the endpoints and intersection points for visualization
        end_dilated = dilation(end, square(15))
        inter_dilated = dilation(inter, square(15))

        median_tor, length, chord, arc, connection_dico = geometricalVBMs.compute_tortuosity_length(binary_array)
        tor_index = np.sum(arc) / np.sum(chord)

        p, border_matrix = geometricalVBMs.compute_perimeter(segmentation2d)

        mean_ba, std_ba, median_ba, angle_dico, centroid = geometricalVBMs.compute_branching_angles(binary_array)

        n_obtuse = sum(1 for angle in angle_dico.values() if angle > 90)
        n_acute = sum(1 for angle in angle_dico.values() if angle < 90)

        # Instantiate fractal VBM object
        from PVBM.FractalAnalysis import MultifractalVBMs
        fractalVBMs = MultifractalVBMs(n_rotations=25, optimize=True, min_proba=0.0001, maxproba=0.9999)

        from scipy.ndimage import zoom

        segmentation2d = (skeleton> 0).astype(np.float64)
        
        # Scale factor for resizing to (2048, 2048)
        scale_factors = (2048 / segmentation2d.shape[0], 2048 / segmentation2d.shape[1])
        
        # Ensure the resized array is strictly binary 
        segmentation2d = zoom(binary_array, scale_factors, order=0)  # order=0 for nearest-neighbor interpolation
        
        segmentation2d=segmentation2d.astype(np.float64)
        
        D0,D1,D2,SL = fractalVBMs.compute_multifractals(segmentation2d)


        image_params_dict[image_file] = {
            'area': a,
            'endpoints': n_end,
            'intersection_points': n_inter,
            'median_tortuosity': median_tor,
            'tortuosity_index': tor_index,
            'length': length,
            'perimeter': p,
            'mean_branching_angles': mean_ba,
            'std_branching_angles': std_ba,
            'median_branching_angles': median_ba,
            'n_obtuse': n_obtuse,
            'n_acute': n_acute,
            'D0': D0,
            'D1': D1,
            'D2': D2,
            'SL': SL,
        }

        # Save parameters after processing each image
        save_parameters_to_file(image_params_dict, category_name)

    return image_params_dict

# Initialize dictionary to store numerical values for each category
image_params_dict = {
#    'A': process_images(category_A, "Category A"),
#    'N': process_images(category_N, "Category N")
#    'G': process_images(category_G, "Category G")
    'D': process_images(category_D, "Category D")
}

# Extract and fit distributions for each numerical parameter
numerical_params = [
    'area', 'endpoints', 'intersection_points', 'median_tortuosity', 'tortuosity_index',
    'length', 'perimeter', 'mean_branching_angles', 'std_branching_angles',
    'median_branching_angles', 'n_obtuse', 'n_acute', 'D0', 'D1', 'D2', 'SL'
]

def overlay_best_fit(param_values, param_name, category):
    plt.figure()
    # Plot the histogram of the data
    count, bins, ignored = plt.hist(param_values, bins=50, alpha=0.6, density=True, label='Data Histogram')

    # Use Fitter to fit the data
    f = Fitter(param_values, distributions=['norm', 'lognorm', 'beta', 'burr', 'gamma'])
    f.fit()
    best_fit = f.get_best(method='sumsquare_error')

    # Get the name of the best distribution and its parameters
    best_fit_name = list(best_fit.keys())[0]
    best_fit_params = f.fitted_param[best_fit_name]

    # Generate points on the x axis for the fitted PDF
    x = np.linspace(min(param_values), max(param_values), 1000)

    # Select the PDF based on the best fit distribution name
    if best_fit_name == 'norm':
        pdf = norm.pdf(x, *best_fit_params)
    elif best_fit_name == 'lognorm':
        s, loc, scale = best_fit_params  # unpack parameters
        pdf = lognorm.pdf(x, s, loc, scale)
    elif best_fit_name == 'beta':
        a, b, loc, scale = best_fit_params
        pdf = beta.pdf(x, a, b, loc, scale)
    elif best_fit_name == 'burr':
        c, d, loc, scale = best_fit_params
        pdf = burr.pdf(x, c, d, loc, scale)
    elif best_fit_name == 'gamma':
        a, loc, scale = best_fit_params
        pdf = gamma.pdf(x, a, loc, scale)

    # Plot the PDF of the best fit distribution
    plt.plot(x, pdf, 'k-', linewidth=2, label=f'Fit: {best_fit_name}')
    plt.xlabel(param_name)
    plt.ylabel('Density')
    plt.title(f'Overlay of Best Fit Distribution on Histogram of {param_name} for {category}')
    plt.legend()
   
    
    # Save the plot to a file
    plot_filename = f"{category}_{param_name}_best_fit.png"
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()

    

# Apply to each numerical parameter for each category
for param in numerical_params:
    for category, images in image_params_dict.items():
        param_values = [image_params[param] for image_params in images.values() if param in image_params]
        print(f"Fitting distributions for {param} in category {category}")
        overlay_best_fit(param_values, param, category)

