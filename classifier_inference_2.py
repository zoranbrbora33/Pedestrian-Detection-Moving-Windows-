import joblib
import yaml
import cv2
from skimage.feature import hog
from nms import Detection, NoneMaximumSuppression

# Instantiate Non-Maximum Suppression object
nms = NoneMaximumSuppression()

# Path to the trained pedestrian classifier model
MODEL_PATH = "pedestrian_classifier.joblib"
# Path to the model configuration file
CFG_PATH = "model_cfg.yaml"
# Path to the test image
TEST_IMAGE_PATH1 = "dataset/multiple_pedestrians.png"
# Sliding window step size
SLIDING_WINDOW_STEP = 25

# Load the trained pedestrian classifier model
pedestrian_classifier = joblib.load(MODEL_PATH)

# Load the model configuration from the YAML file
with open(CFG_PATH, "r") as file:
    model_cfg_dict = yaml.safe_load(file)

# Retrieve mean pedestrian height and width from the model configuration
mean_pedestrian_height = model_cfg_dict["MEAN PEDESTRIAN HEIGHT"]
mean_pedestrian_width = model_cfg_dict["MEAN PEDESTRIAN WIDTH"]

# Read the test image
test_image = cv2.imread(TEST_IMAGE_PATH1)
# Convert the test image to grayscale
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Get the height and width of the test image
img_height, img_width = test_image_gray.shape

print("Detection started")
# Slide a window over the test image with the specified step size
for i in range(0, img_height - mean_pedestrian_height, SLIDING_WINDOW_STEP):
    for j in range(0, img_width - mean_pedestrian_width, SLIDING_WINDOW_STEP):
        # Calculate the top-left and bottom-right bounding box coordinates
        top_left_bb = (j, i)
        bottom_right_bb = (j + mean_pedestrian_width,
                           i + mean_pedestrian_height)

        # Extract the region of interest (ROI) from the grayscale image
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

        if is_pedestrian_probability > 0.9:
            # Create a Detection object with the probability and bounding box coordinates
            detection = Detection(is_pedestrian_probability,
                                  (top_left_bb, bottom_right_bb))

            # Add the detection to the Non-Maximum Suppression object
            nms_object = nms.add_detection(detection=detection)

print("Detection finished")

# Apply Non-Maximum Suppression to filter out redundant detections
detection = nms.apply_nms()

# Iterate over the filtered detections
for pedestrian in detection:
    top_left, bottom_right = pedestrian.rect_coords

    # Draw a rectangle around each detected pedestrian in the final image
    cv2.rectangle(test_image, top_left, bottom_right, (0, 255, 0), thickness=1)

# Display the final image with the bounding boxes
cv2.imshow("Final", test_image)
cv2.waitKey()
cv
