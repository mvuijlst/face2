import os
import cv2
import dlib
import numpy as np
from imutils import face_utils

# Function to load all images from a folder
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):  # Add more extensions if needed
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images

# Initialize dlib's face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download from dlib

# Placeholder for manual eye selection
selected_points = []
manual_mode = False
manual_frame = None

# Mouse callback for selecting eye points manually
def select_eyes(event, x, y, flags, param):
    global selected_points, manual_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        # Record point
        selected_points.append((x, y))
        # Visual feedback of the selected points
        cv2.circle(manual_frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Eyes", manual_frame)

        # If two points (eyes) are selected, we proceed
        if len(selected_points) == 2:
            cv2.destroyWindow("Select Eyes")

# Function to detect landmarks
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) > 0:
        return predictor(gray, faces[0])  # Return landmarks for the first detected face
    return None

# Function to align the face
def align_face(image, landmarks=None, manual_points=None):
    if landmarks is not None:
        landmarks = face_utils.shape_to_np(landmarks)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
    elif manual_points is not None:
        left_eye_center = np.array(manual_points[0])
        right_eye_center = np.array(manual_points[1])
    else:
        return None

    if landmarks is not None:
        left_eye_center = left_eye.mean(axis=0).astype("int")
        right_eye_center = right_eye.mean(axis=0).astype("int")

    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx)) - 180
    dist = np.sqrt((dx ** 2) + (dy ** 2))
    desired_dist = 100  # Desired distance between eyes
    scale = desired_dist / dist

    eyes_center = (float((left_eye_center[0] + right_eye_center[0]) // 2),
                   float((left_eye_center[1] + right_eye_center[1]) // 2))

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    output = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    return output

# Main loop for loading images and selecting points manually if necessary
def process_images(images):
    aligned_images = []
    global manual_frame, selected_points, manual_mode

    for img in images:
        landmarks = get_landmarks(img)
        if landmarks is None:
            # If no landmarks, switch to manual mode
            manual_frame = img.copy()
            selected_points = []
            cv2.imshow("Select Eyes", manual_frame)
            cv2.setMouseCallback("Select Eyes", select_eyes)
            cv2.waitKey(0)  # Wait until points are selected
            aligned_img = align_face(img, manual_points=selected_points)
        else:
            aligned_img = align_face(img, landmarks=landmarks)

        if aligned_img is not None:
            aligned_images.append(aligned_img)

    return aligned_images

# Function to generate an MP4 video
def create_video(aligned_images, output_path='output_video.mp4', fps=30):
    height, width, layers = aligned_images[0].shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in aligned_images:
        video.write(image)

    video.release()

# Example usage
folder_path = "photos"  # Replace with the folder containing your images
images = load_images(folder_path)  # Load images
aligned_images = process_images(images)  # Align images

# Create MP4 video from aligned images
create_video(aligned_images, output_path='output_animation.mp4')
print("MP4 video created!")
