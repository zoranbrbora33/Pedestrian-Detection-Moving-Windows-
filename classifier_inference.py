import joblib
import yaml
import cv2
from skimage.feature import hog

MODEL_PATH = "pedestrian_classifier.joblib"
CFG_PATH = "model_cfg.yaml"
TEST_IMAGE_PATH = "dataset/test_img.png"
SLIDING_WINDOWS_STEP = 25

# Load the pedestrian classifier model
pedestrian_classifier = joblib.load(MODEL_PATH)

# Load the model configuration from YAML file
with open(CFG_PATH, "r") as file:
    model_cfg_dict = yaml.safe_load(file)

# Retrieve the mean pedestrian height and width from the configuration
mean_pedestrian_height = model_cfg_dict["MEAN PEDESTRIAN HEIGHT"]
mean_pedestrian_width = model_cfg_dict["MEAN PEDESTRIAN WIDTH"]

# Load the test image
test_image = cv2.imread(TEST_IMAGE_PATH)
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Get the height and width of the test image
image_height, image_width = test_image_gray.shape

# Initialize variables to track maximum pedestrian probability and final bounding box coordinates
max_pedestrian_probability = 0
final_top_left_bb = (0, 0)

# Sliding window approach to scan the image
for i in range(0, image_height - mean_pedestrian_height, SLIDING_WINDOWS_STEP):
    for j in range(0, image_width - mean_pedestrian_width, SLIDING_WINDOWS_STEP):
        # Define the top-left corner of the current sliding window
        top_left_bb = (j, i)
        # Extract the region of interest (ROI) from the test image based on the sliding window position
        roi = test_image_gray[i:i+mean_pedestrian_height,
                              j:j+mean_pedestrian_width]
        # Compute HOG descriptor for the ROI
        HOG_desc, hog_image = hog(roi, visualize=True)
        HOG_desc = HOG_desc.reshape((1, -1))
        # Use the pedestrian classifier to predict the pedestrian probabilities for the ROI
        roi_pedestrian_probabilities = pedestrian_classifier.predict_proba(
            HOG_desc)
        # Extract the probability of being a pedestrian (class 1)
        is_pedestrian_proba = roi_pedestrian_probabilities[0, 1]

        # Update the maximum pedestrian probability and final bounding box coordinates if a higher probability is found
        if is_pedestrian_proba > max_pedestrian_probability:
            max_pedestrian_probability = is_pedestrian_proba
            final_top_left_bb = top_left_bb

print("Max pedestrian probability: {}".format(max_pedestrian_probability))

# Draw the final bounding box on the test image
cv2.rectangle(test_image, final_top_left_bb,
              (final_top_left_bb[0] + mean_pedestrian_width,
               final_top_left_bb[1] + mean_pedestrian_height),
              (0, 255, 0), thickness=3)

# Display the image with the final detection
cv2.imshow("Final detection", test_image)
cv2.waitKey()
cv2.destroyAllWindows()
