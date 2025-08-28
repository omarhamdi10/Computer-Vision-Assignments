from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the histogram of a grayscale image
def calculate_histogram(image_array):
    histogram = np.zeros(256, dtype=int)
    for pixel_value in image_array.flatten():
        histogram[pixel_value] += 1
    return histogram

# Function to calculate the cumulative histogram
def calculate_cumulative_histogram(histogram):
    cumulative_histogram = np.zeros_like(histogram)
    cumulative_histogram[0] = histogram[0]
    for i in range(1, len(histogram)):
        cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i]
    return cumulative_histogram

# Function to get color intensities at a specified percentage
def get_color_at_percentage(cumulative_histogram, percentage):
    total_pixels = cumulative_histogram[-1]
    lower_bound = total_pixels * (percentage / 100.0)
    upper_bound = total_pixels * (1 - (percentage / 100.0))

    min_intensity = next(i for i, count in enumerate(cumulative_histogram) if count >= lower_bound)
    max_intensity = next(i for i, count in enumerate(cumulative_histogram) if count >= upper_bound)
    
    return min_intensity, max_intensity

# Function to get color intensities with the maximum slope
def get_colors_at_max_slope(cumulative_histogram):
    max_slope = 0
    min_intensity = 0
    max_intensity = 0

    for i in range(0, 255):
        for j in range(i + 1, 256):
            slope = (cumulative_histogram[j] - cumulative_histogram[i]) / (j - i)
            if slope > max_slope:
                max_slope = slope
                min_intensity = i
                max_intensity = j

    return min_intensity, max_intensity

# Function to apply contrast stretching
def stretch_contrast(image_array, in_min, in_max, out_min=0, out_max=255):
    stretched_image = (image_array.astype(float) - in_min) * ((out_max - out_min) / (in_max - in_min)) + out_min
    stretched_image[stretched_image < 0] = 0  # Set any negative values to zero
    stretched_image = np.clip(stretched_image, out_min, out_max)  # Clip values to the specified range
    return stretched_image.astype(np.uint8)

# Function to apply histogram equalization for the full range
def equalize_histogram(image_array):
    histogram = calculate_histogram(image_array)
    cumulative_histogram = calculate_cumulative_histogram(histogram)
    
    total_pixels = cumulative_histogram[-1]
    equalized_image = (cumulative_histogram[image_array] - cumulative_histogram.min()) * 255 / (total_pixels - cumulative_histogram.min())
    
    return equalized_image.astype(np.uint8)

# Function to apply histogram equalization within a specified intensity range
def equalize_histogram_with_range(image_array, min_intensity, max_intensity):
    histogram = calculate_histogram(image_array)
    cumulative_histogram = calculate_cumulative_histogram(histogram)
    total_pixels = cumulative_histogram[-1]
    scaled_cumulative = (cumulative_histogram - cumulative_histogram[min_intensity]) * 255 / (cumulative_histogram[max_intensity] - cumulative_histogram[min_intensity])
    equalized_image = scaled_cumulative[image_array]
    return equalized_image.astype(np.uint8)

# Function to plot the image and its histograms
def plot_image_and_histograms(image_array, title, min_intensity=None, max_intensity=None):
    histogram = calculate_histogram(image_array)
    cumulative_histogram = calculate_cumulative_histogram(histogram)
    
    # Plot the image
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image_array, cmap='gray')
    plt.title(title)
    plt.axis('off')

    # Plot the histogram
    plt.subplot(1, 3, 2)
    plt.bar(range(256), histogram, color='blue')
    if min_intensity is not None and max_intensity is not None:
        plt.axvline(min_intensity, color='red', linestyle='--', label=f'Min at {min_intensity}')
        plt.axvline(max_intensity, color='green', linestyle='--', label=f'Max at {max_intensity}')
    plt.title(f'Histogram for {title}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot the cumulative histogram
    plt.subplot(1, 3, 3)
    plt.plot(range(256), cumulative_histogram, color='green')
    if min_intensity is not None and max_intensity is not None:
        plt.axvline(min_intensity, color='red', linestyle='--', label=f'Min at {min_intensity}')
        plt.axvline(max_intensity, color='green', linestyle='--', label=f'Max at {max_intensity}')
    plt.title(f'Cumulative Histogram for {title}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Cumulative Frequency')
    plt.legend()

    plt.show()

# Load and process the image
image_path = r"C:\Users\DELL\Desktop\Test1.png"  # Replace with your actual image path
image = Image.open(image_path).convert('L')
image_array = np.array(image)

# Calculate histogram and cumulative histogram
histogram = calculate_histogram(image_array)
cumulative_histogram = calculate_cumulative_histogram(histogram)

# Run evaluation for Analyze Histogram
min_5, max_5 = get_color_at_percentage(cumulative_histogram, 5)
min_10, max_10 = get_color_at_percentage(cumulative_histogram, 10)
min_15, max_15 = get_color_at_percentage(cumulative_histogram, 15)
min_slope, max_slope = get_colors_at_max_slope(cumulative_histogram)

# Display original image and histograms with different min/max intensities
plot_image_and_histograms(image_array, "Original Image")

# Apply and display contrast stretching for 5%, 10%, 15% and max slope
stretched_5 = stretch_contrast(image_array, min_5, max_5)
plot_image_and_histograms(stretched_5, "Contrast Stretched (5%)", min_5, max_5)

stretched_10 = stretch_contrast(image_array, min_10, max_10)
plot_image_and_histograms(stretched_10, "Contrast Stretched (10%)", min_10, max_10)

stretched_15 = stretch_contrast(image_array, min_15, max_15)
plot_image_and_histograms(stretched_15, "Contrast Stretched (15%)", min_15, max_15)

stretched_slope = stretch_contrast(image_array, min_slope, max_slope)
plot_image_and_histograms(stretched_slope, "Contrast Stretched (Max Slope)", min_slope, max_slope)

# Apply and display full-range histogram equalization
equalized_image = equalize_histogram(image_array)
plot_image_and_histograms(equalized_image, "Full Histogram Equalized Image")

# Apply and display histogram equalization for each case
equalized_5 = equalize_histogram_with_range(image_array, min_5, max_5)
plot_image_and_histograms(equalized_5, "Histogram Equalized (5%)", min_5, max_5)

equalized_10 = equalize_histogram_with_range(image_array, min_10, max_10)
plot_image_and_histograms(equalized_10, "Histogram Equalized (10%)", min_10, max_10)

equalized_15 = equalize_histogram_with_range(image_array, min_15, max_15)
plot_image_and_histograms(equalized_15, "Histogram Equalized (15%)", min_15, max_15)
