import cv2
import numpy as np

def evaluate_saliency(image):
    # Load the image
    image = cv2.imread('output/Highway_billboard_region_depth.png')

    # Initialize OpenCV's static saliency spectral residual detector and compute the saliency map
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (_, saliencyMap) = saliency.computeSaliency(image)

    # If we want the saliency map to be in range [0, 255], we can convert it
    saliencyMap = (saliencyMap * 255).astype("uint8")

    # Calculate the mean and standard deviation of the saliency map
    mean = np.mean(saliencyMap)
    std = np.std(saliencyMap)

    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")

    # Display the image and the saliency map
    cv2.imshow("Image", image)
    cv2.imshow("Output", saliencyMap)
    cv2.waitKey(0)

def main(image, top_left, bottom_right):
    from PIL import Image
    import numpy as np

    # Open the image file
    img = Image.open(image)

    # Convert the image data to a numpy array
    img_data = np.array(img)

    # Slice the numpy array to get the ROI
    roi_data = img_data[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    # Compute the mean and standard deviation of the ROI
    mean_roi = np.mean(roi_data)
    std_roi = np.std(roi_data)

    print(f'Mean of ROI: {mean_roi}')
    print(f'Standard Deviation of ROI: {std_roi}')

if __name__ == '__main__':
    main('output/Highway_billboard_region_depth.png', (172, 186), (186, 210))
    # main('output/Highway_billboard_region_depth.png', (59, 72), (114, 164))