import os
import cv2
import dlib
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps
import json
from imutils import face_utils

# Initialize Dlib's face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

CONFIG_FILE = "app_config.json"
EYE_POSITIONS_FILE = "eye_positions.json"

# Helper function to load configuration from JSON
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            return json.load(file)
    return {}

# Helper function to save configuration to JSON
def save_config(config):
    with open(CONFIG_FILE, 'w') as file:
        json.dump(config, file)

# Helper function to load saved eye positions from JSON
def load_eye_positions():
    if os.path.exists(EYE_POSITIONS_FILE):
        with open(EYE_POSITIONS_FILE, 'r') as file:
            return json.load(file)
    return {}

# Helper function to save eye positions to JSON
def save_eye_positions(eye_positions):
    with open(EYE_POSITIONS_FILE, 'w') as file:
        serializable_eye_positions = {
            filename: [[int(pupil[0]), int(pupil[1])] for pupil in pupils]
            for filename, pupils in eye_positions.items()
        }
        json.dump(serializable_eye_positions, file)


# Load all images from a folder and sort by modified date
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            file_path = os.path.join(folder, filename)
            img = cv2.imread(file_path)
            if img is not None:
                mod_time = os.path.getmtime(file_path)
                images.append((filename, img, mod_time))
            else:
                print(f"Skipping file {filename} due to loading error.")
    images.sort(key=lambda x: x[2])
    return images

# Detect pupils in the largest face
def detect_pupils_with_preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 0:
        return (0, 0), (0, 0)

    largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
    landmarks = predictor(gray, largest_face)
    landmarks = face_utils.shape_to_np(landmarks)

    # Extract the eye landmarks
    left_eye_landmarks = landmarks[36:42]
    right_eye_landmarks = landmarks[42:48]

    # Extract and preprocess eye regions
    left_eye_region = extract_eye_region(gray, left_eye_landmarks)
    right_eye_region = extract_eye_region(gray, right_eye_landmarks)

    left_eye_region = preprocess_eye(left_eye_region)
    right_eye_region = preprocess_eye(right_eye_region)

    # Detect pupils
    left_pupil = detect_pupil_in_eye(left_eye_region)
    right_pupil = detect_pupil_in_eye(right_eye_region)

    # Convert to global coordinates
    left_pupil_global = convert_to_global_coords(left_pupil, left_eye_landmarks)
    right_pupil_global = convert_to_global_coords(right_pupil, right_eye_landmarks)

    return left_pupil_global, right_pupil_global

def extract_eye_region(gray_image, eye_landmarks):
    x_min = min(eye_landmarks[:, 0])
    x_max = max(eye_landmarks[:, 0])
    y_min = min(eye_landmarks[:, 1])
    y_max = max(eye_landmarks[:, 1])
    return gray_image[y_min:y_max, x_min:x_max]

def preprocess_eye(eye_region):
    eye_preprocessed = cv2.equalizeHist(eye_region)
    return eye_preprocessed

def detect_pupil_in_eye(eye_region):
    _, thresh = cv2.threshold(eye_region, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return (0, 0)
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M['m00'] == 0:
        return (0, 0)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)

def convert_to_global_coords(pupil_position, eye_landmarks):
    x_min = min(eye_landmarks[:, 0])
    y_min = min(eye_landmarks[:, 1])
    global_x = pupil_position[0] + x_min
    global_y = pupil_position[1] + y_min
    return (global_x, global_y)

# GUI Application for reviewing and correcting eye detection
class EyeReviewApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Eye Detection Review")

        # Load window position and size
        config = load_config()
        if 'window_size' in config:
            self.master.geometry(config['window_size'])

        # Load previously selected folder from config
        config = load_config()
        self.folder_path = config.get('last_folder', None)

        if not self.folder_path or not os.path.exists(self.folder_path):
            # Ask for folder if none saved or folder doesn't exist
            self.folder_path = filedialog.askdirectory(title="Select Folder with Images")
            if not self.folder_path:
                messagebox.showerror("Error", "No folder selected!")
                self.master.quit()

        # Load images and eye positions
        self.images = load_images(self.folder_path)
        self.eye_positions = load_eye_positions()

        self.detection_results = [None] * len(self.images)
        self.manual_corrections = [False] * len(self.images)
        self.current_image_index = None
        self.clicks = []

        # Left pane - Listbox for image list with detection status
        self.listbox_frame = tk.Frame(master)
        self.listbox_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        self.listbox = tk.Listbox(self.listbox_frame, width=30, height=20)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_image_select)

        # Right pane - Canvas for displaying images
        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, width=600, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Populate listbox with file names
        self.populate_listbox()

        # Start background processing for pupil detection
        threading.Thread(target=self.process_images_in_background, daemon=True).start()

        # Save window size and position on close
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    # Populate the listbox with file names (initially without detection status)
    def populate_listbox(self):
        for filename, _, _ in self.images:
            if filename in self.eye_positions:
                self.listbox.insert(tk.END, f"{filename} ⚙️")
            else:
                self.listbox.insert(tk.END, f"{filename} ⏳")

    # Background thread to process images for pupil detection
    def process_images_in_background(self):
        for idx, (filename, image, _) in enumerate(self.images):
            if filename not in self.eye_positions:
                left_pupil, right_pupil = detect_pupils_with_preprocessing(image)
                
                if left_pupil != (0, 0) and right_pupil != (0, 0):
                    self.eye_positions[filename] = [left_pupil, right_pupil]
                    save_eye_positions(self.eye_positions)
                    self.update_listbox_item(idx, filename, True)
                else:
                    self.update_listbox_item(idx, filename, False)

    # Update a specific item in the listbox (success or failure)
    def update_listbox_item(self, index, filename, success):
        status_icon = "✅" if success else "❌"
        new_text = f"{filename} {status_icon}"
        self.listbox.delete(index)
        self.listbox.insert(index, new_text)

    # Load and display selected image on the right pane
    def on_image_select(self, event):
        selection = event.widget.curselection()
        if not selection:
            return

        index = selection[0]
        self.current_image_index = index
        filename, image, _ = self.images[index]
        pupil_positions = self.eye_positions.get(filename, None)

        self.display_image(image, pupil_positions)

    # Display image on canvas and draw pupils
    def display_image(self, image, pupil_positions):
        original_height, original_width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        resized_pil_image, scale_factor_w, scale_factor_h = self.resize_image(pil_image, self.canvas.winfo_width(), self.canvas.winfo_height())
        self.current_image = ImageTk.PhotoImage(resized_pil_image)

        # Clear previous drawings
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)

        if pupil_positions:
            left_pupil, right_pupil = pupil_positions
            left_pupil_scaled = (int(left_pupil[0] * scale_factor_w), int(left_pupil[1] * scale_factor_h))
            right_pupil_scaled = (int(right_pupil[0] * scale_factor_w), int(right_pupil[1] * scale_factor_h))

            self.canvas.create_oval(left_pupil_scaled[0] - 2, left_pupil_scaled[1] - 2,
                                    left_pupil_scaled[0] + 2, left_pupil_scaled[1] + 2,
                                    outline="yellow", width=2)
            self.canvas.create_oval(right_pupil_scaled[0] - 2, right_pupil_scaled[1] - 2,
                                    right_pupil_scaled[0] + 2, right_pupil_scaled[1] + 2,
                                    outline="yellow", width=2)

    # Resize image to fit the canvas while keeping the aspect ratio
    def resize_image(self, image, canvas_width, canvas_height):
        image_width, image_height = image.size
        ratio = min(canvas_width / image_width, canvas_height / image_height)
        new_size = (int(image_width * ratio), int(image_height * ratio))
        resized_image = image.resize(new_size, Image.LANCZOS)
        return resized_image, ratio, ratio

    # Handle mouse click on canvas for manual eye selection
    def on_canvas_click(self, event):
        if self.current_image_index is None:
            return

        clicked_x = event.x
        clicked_y = event.y
        _, scale_factor_w, scale_factor_h = self.resize_image(Image.fromarray(self.images[self.current_image_index][1]),
                                                              self.canvas.winfo_width(), self.canvas.winfo_height())

        original_x = int(clicked_x / scale_factor_w)
        original_y = int(clicked_y / scale_factor_h)
        self.clicks.append((original_x, original_y))

        if len(self.clicks) == 2:
            left_pupil, right_pupil = self.clicks
            self.eye_positions[self.images[self.current_image_index][0]] = [left_pupil, right_pupil]
            self.display_image(self.images[self.current_image_index][1], self.eye_positions[self.images[self.current_image_index][0]])
            save_eye_positions(self.eye_positions)
            self.update_listbox_item(self.current_image_index, self.images[self.current_image_index][0], True)
            self.clicks = []

    # Handle window close to save size, position, and last folder
    def on_close(self):
        window_geometry = self.master.geometry()
        config = load_config()
        config['window_size'] = window_geometry
        config['last_folder'] = self.folder_path
        save_config(config)
        self.master.destroy()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = EyeReviewApp(root)
    root.mainloop()
