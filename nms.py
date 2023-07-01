class NoneMaximumSuppression:
    def __init__(self):
        self.__detections_before_nms = []
        self.__detections_after_nms = []

    def add_detection(self, detection):
        """
        Add a detection to the list before applying non-maximum suppression.

        Args:
            detection (Detection): The detection object to add.

        """
        self.__detections_before_nms.append(detection)

    def __calculate_iou(self, box_a, box_b):
        """
        Calculate the intersection over union (IoU) between two bounding boxes.

        Args:
            box_a (tuple): Coordinates of the first bounding box ((x1, y1), (x2, y2)).
            box_b (tuple): Coordinates of the second bounding box ((x1, y1), (x2, y2)).

        Returns:
            float: The IoU value.

        """
        # Extract the coordinates of the bounding boxes
        xA = max(box_a[0][0], box_b[0][0])
        yA = max(box_a[0][1], box_b[0][1])
        xB = min(box_a[1][0], box_b[1][0])
        yB = min(box_a[1][1], box_b[1][1])

        # Calculate the intersection area
        intersection = max(0, xB - xA) * max(0, yB - yA)
        # Calculate the area of box_a
        box_a_area = (box_a[1][0] - box_a[0][0]) * (box_a[1][1] - box_a[0][1])
        # Calculate the area of box_b
        box_b_area = (box_b[1][0] - box_b[0][0]) * (box_b[1][1] - box_b[0][1])

        # Calculate the union area
        union = box_a_area + box_b_area - intersection
        # Calculate the IoU value
        iou = intersection / union

        return iou

    def apply_nms(self, iou_threshold=0.2):
        """
        Apply non-maximum suppression to the list of detections.

        Args:
            iou_threshold (float): The IoU threshold for considering two detections as the same object.

        Returns:
            list: The list of detections after non-maximum suppression.

        """
        # Sort the detections in descending order based on their probabilities
        self.__detections_before_nms.sort(
            key=lambda x: x.probability, reverse=True)

        removed_detection_indices = []

        for i, detection in enumerate(self.__detections_before_nms):
            # Skip this detection if it has been removed in a previous iteration
            if i in removed_detection_indices:
                continue

            # Add the detection to the list after non-maximum suppression
            self.__detections_after_nms.append(detection)

            for j in range(i + 1, len(self.__detections_before_nms)):
                box_a = detection.rect_coords
                box_b = self.__detections_before_nms[j].rect_coords
                iou = self.__calculate_iou(box_a, box_b)

                # Check if the IoU is above the threshold
                if iou > iou_threshold:
                    # Mark the detection for removal
                    removed_detection_indices.append(j)

        return self.__detections_after_nms


class Detection:
    def __init__(self, probability, rect_coords):
        """
        Initialize a Detection object.

        Args:
            probability (float): The probability of the detection.
            rect_coords (tuple): The coordinates of the bounding box ((x1, y1), (x2, y2)).

        """
        self.probability = probability
        self.rect_coords = rect_coords
