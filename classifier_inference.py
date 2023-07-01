import joblib
import yaml
import cv2
from skimage.feature import hog

# Path to the trained pedestrian classifier model
MODEL_PATH = "pedestrian_classifier.joblib"

# Path to the model configuration file
CFG_PATH = "model_cfg.yaml"

# Path to the test image
TEST_IMAGE_PATH = "dataset/no_pedestrian.jpg"

# Sliding window step size
SLIDING_WINDOW_STEP = 20

# Load the trained pedestrian classifier model
pedestrian_classifier = joblib.load(MODEL_PATH)

# Load the model configuration from the YAML file
with open(CFG_PATH, "r") as file:
    model_cfg_dict = yaml.safe_load(file)

# Retrieve mean pedestrian height and width from the model configuration
mean_pedestrian_height = model_cfg_dict["MEAN PEDESTRIAN HEIGHT"]
mean_pedestrian_width = model_cfg_dict["MEAN PEDESTRIAN WIDTH"]

# Read the test image
test_image = cv2.imread(TEST_IMAGE_PATH)

# Convert the test image to grayscale
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Get the height and width of the test image
img_height, img_width = test_image_gray.shape

# Variables to store the maximum pedestrian probability and top left bounding box coordinates
max_pedestrian_probability = 0
final_top_left_bb = (0, 0)

# List to store pedestrian probabilities for each sliding window position
pedestrian_probability_list = []

print("Detection started")

# Slide a window over the test image with the specified step size
for i in range(0, img_height - mean_pedestrian_height, SLIDING_WINDOW_STEP):
    for j in range(0, img_width - mean_pedestrian_width, SLIDING_WINDOW_STEP):
        # Calculate the top left bounding box coordinates
        top_left_bb = (j, i)

        # Extract the region of interest (ROI) from the test image
        roi = test_image_gray[i: i + mean_pedestrian_height,
                              j: j + mean_pedestrian_width]

        # Compute Histogram of Oriented Gradients (HOG) features for the ROI
        HOG_desc, hog_img = hog(roi, visualize=True)

        # Reshape HOG features to match the expected input shape of the classifier
        HOG_desc = HOG_desc.reshape(1, -1)

        # Predict the pedestrian probabilities for the ROI using the trained classifier
        roi_pedestrian_probabilities = pedestrian_classifier.predict_proba(
            HOG_desc)

        # Retrieve the probability of the ROI containing a pedestrian
        is_pedestrian_probability = roi_pedestrian_probabilities[0, 1]

        # Add the pedestrian probability and top left bounding box coordinates to the list
        # pedestrian_probability_list.append([is_pedestrian_probability, top_left_bb])

        # Update the maximum pedestrian probability and top left bounding box if a higher probability is found
        if is_pedestrian_probability > max_pedestrian_probability:
            max_pedestrian_probability = is_pedestrian_probability
            final_top_left_bb = top_left_bb

print("Detection finished")
print("Max pedestrian probability: {} ".format(max_pedestrian_probability))

# If the maximum pedestrian probability is above a certain threshold, draw a bounding box around the detected pedestrian
if max_pedestrian_probability > 0.7:
    cv2.rectangle(test_image, final_top_left_bb,
                  ((final_top_left_bb[0] + mean_pedestrian_width),
                   (final_top_left_bb[1] + mean_pedestrian_height)),
                  (0, 255, 0),
                  thickness=3)

# Display the final image with the bounding box
cv2.imshow("Final", test_image)
cv2.waitKey()
cv2.destroyAllWindows()
