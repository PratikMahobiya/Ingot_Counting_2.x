import tensorflow as tf
import csv
import cv2
import numpy as np
from utils import visualization_utils as vis_util


def object_counting_webcam(detect_fn, category_index, is_color_recognition_enabled):

        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        width_heigh_taken = True
        height = 0
        width = 0

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        cap = cv2.VideoCapture(0)
        (ret, frame) = cap.read()

        # for all the frames that are extracted from input video
        while True:
            # Capture frame-by-frame
            (ret, frame) = cap.read()          

            if not  ret:
                print("end of the video file...")
                break
            
            input_frame = frame

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(input_frame, axis=0)

            # insert information text to video frame
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Visualization of the results of a detection.        
            counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                  input_frame,
                                                                                                  1,
                                                                                                  is_color_recognition_enabled,
                                                                                                  np.squeeze(boxes),
                                                                                                  np.squeeze(classes).astype(np.int32),
                                                                                                  np.squeeze(scores),
                                                                                                  category_index,
                                                                                                  use_normalized_coordinates=True,
                                                                                                  line_thickness=4)
            if(len(counting_mode) == 0):
                cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
            else:
                cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
            
            cv2.imshow('object counting',input_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
