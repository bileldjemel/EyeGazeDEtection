import cv2
import dlib
import numpy as np
import time
import os

# Initialize face and landmark detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize video capture object
cap = cv2.VideoCapture(0)

# Create a folder to save fixation data
folder_path = "FixationData"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Open the file for writing eye positions
file = open(os.path.join(folder_path, "eye_positions.txt"), "w")

# Open the file for writing fixation information
fixation_file = open(os.path.join(folder_path, "fixation_info.txt"), "w")

# Initialize variables for I-DT algorithm
prev_fixation = None
fixation_count = 0
fixation_threshold = 5  # Adjust this value based on your requirement

# Initialize variables for fixation duration calculation
fixation_start_time = None
fixation_duration_threshold = 0.5  # Adjust this value based on your requirement (in seconds)

while True:
    ret, frame = cap.read()

    # Perform face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Iterate over detected faces
    for face in faces:
        landmarks = predictor(gray, face)

        # Extract eye landmarks
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        # Calculate the average position of the left and right eye
        left_eye_center = np.mean(left_eye, axis=0).astype(int)
        right_eye_center = np.mean(right_eye, axis=0).astype(int)

        # Apply I-DT algorithm
        if prev_fixation is None:
            # Initialize the first fixation
            prev_fixation = (left_eye_center[0], left_eye_center[1])
            fixation_count = 1
            fixation_start_time = time.time()
        else:
            # Calculate the Euclidean distance between current eye position and previous fixation
            distance = np.linalg.norm(np.array([left_eye_center[0], left_eye_center[1]]) - np.array(prev_fixation))

            if distance < fixation_threshold:
                # Eye position is within the fixation threshold
                fixation_count += 1
            else:
                # Eye position is outside the fixation threshold
                if fixation_count > 5:  # Adjust this value based on your requirement
                    # Save the fixation position
                    fixation_file.write(f"Fixation Position: {prev_fixation[0]}, {prev_fixation[1]}\n")
                    # Calculate fixation duration
                    fixation_end_time = time.time()
                    fixation_duration = fixation_end_time - fixation_start_time
                    if fixation_duration >= fixation_duration_threshold:
                        # Save the fixation duration
                        fixation_file.write(f"Fixation Duration: {fixation_duration:.2f} seconds\n")
                # Reset the fixation count and update the previous fixation
                fixation_count = 1
                prev_fixation = (left_eye_center[0], left_eye_center[1])
                fixation_start_time = time.time()

        # Save eye positions to the file
        file.write(f"Left Eye: {left_eye_center[0]}, {left_eye_center[1]}\n")
        file.write(f"Right Eye: {right_eye_center[0]}, {right_eye_center[1]}\n")

        # Plot eye positions on the frame
        cv2.circle(frame, tuple(left_eye_center), 2, (0, 0, 255), -1)
        cv2.circle(frame, tuple(right_eye_center), 2, (0, 0, 255), -1)
                   
        # Display eye positions on the screen
        cv2.putText(frame, f"Left Eye: {left_eye_center[0]}, {left_eye_center[1]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Right Eye: {right_eye_center[0]}, {right_eye_center[1]}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the files
file.close()
fixation_file.close()

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

