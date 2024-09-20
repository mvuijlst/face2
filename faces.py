import os
import cv2
import dlib
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ExifTags
import sqlite3
from imutils import face_utils
import platform
import datetime
import tkinter.ttk as ttk  # For progress bar

# Initialize Dlib's face detector and predictor
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
except RuntimeError:
    messagebox.showerror("Error", "Shape predictor file not found!")
    quit()

DB_FILE = "app_data.db"

# Function to extract date/time from EXIF metadata
def get_image_datetime(file_path):
    try:
        img = Image.open(file_path)
        exif_data = img._getexif()
        if exif_data is not None:
            exif = {
                ExifTags.TAGS.get(tag, tag): value
                for tag, value in exif_data.items()
            }
            datetime_original = exif.get('DateTimeOriginal')
            if datetime_original:
                datetime_obj = datetime.datetime.strptime(datetime_original, '%Y:%m:%d %H:%M:%S')
                return datetime_obj
    except Exception as e:
        print(f"Error reading EXIF data from {file_path}: {e}")
    return None

# Initialize SQLite database and create tables
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Create table for configuration
    c.execute('''CREATE TABLE IF NOT EXISTS config
                 (key TEXT PRIMARY KEY, value TEXT)''')

    # Create table for storing eye positions
    c.execute('''CREATE TABLE IF NOT EXISTS eye_positions
                 (filename TEXT PRIMARY KEY,
                  left_pupil_x INTEGER, left_pupil_y INTEGER,
                  right_pupil_x INTEGER, right_pupil_y INTEGER)''')

    # Check if datetime_taken column exists
    c.execute("PRAGMA table_info(eye_positions)")
    columns = [info[1] for info in c.fetchall()]
    if 'datetime_taken' not in columns:
        c.execute("ALTER TABLE eye_positions ADD COLUMN datetime_taken TEXT")

    conn.commit()
    conn.close()

# Helper functions for loading/saving configurations and eye positions
def load_config():
    config = {}
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT key, value FROM config")
    for key, value in c.fetchall():
        config[key] = value
    conn.close()
    return config

def save_config(config):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    for key, value in config.items():
        c.execute("REPLACE INTO config (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def load_eye_positions():
    eye_positions = {}
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''SELECT filename, left_pupil_x, left_pupil_y,
                 right_pupil_x, right_pupil_y, datetime_taken FROM eye_positions''')
    for row in c.fetchall():
        filename, left_x, left_y, right_x, right_y, datetime_taken = row
        # Ensure that coordinates are integers
        left_x = int(left_x)
        left_y = int(left_y)
        right_x = int(right_x)
        right_y = int(right_y)
        # Convert datetime string to datetime object
        if datetime_taken:
            datetime_taken = datetime.datetime.strptime(datetime_taken, '%Y-%m-%d %H:%M:%S')
        else:
            datetime_taken = None
        eye_positions[filename] = {
            'pupils': [(left_x, left_y), (right_x, right_y)],
            'datetime_taken': datetime_taken
        }
    conn.close()
    return eye_positions

def save_eye_positions(filename, left_pupil, right_pupil, datetime_taken):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Store coordinates as integers and date/time as string
    datetime_str = datetime_taken.strftime('%Y-%m-%d %H:%M:%S') if datetime_taken else None
    c.execute('''REPLACE INTO eye_positions (filename, left_pupil_x, left_pupil_y,
                 right_pupil_x, right_pupil_y, datetime_taken)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (filename, int(left_pupil[0]), int(left_pupil[1]),
               int(right_pupil[0]), int(right_pupil[1]), datetime_str))
    conn.commit()
    conn.close()

# Load all images from a folder and sort by date/time taken
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            file_path = os.path.join(folder, filename)
            img_cv = cv2.imread(file_path)
            if img_cv is not None:
                # Extract date/time from EXIF metadata
                datetime_taken = get_image_datetime(file_path)
                # If EXIF data is not available, use file modification time
                if datetime_taken is None:
                    mod_time = os.path.getmtime(file_path)
                    datetime_taken = datetime.datetime.fromtimestamp(mod_time)
                images.append((filename, img_cv, datetime_taken))
            else:
                print(f"Skipping file {filename} due to loading error.")
    # Sort images by date/time taken
    images.sort(key=lambda x: x[2])
    return images

# Pupil detection functions
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

        # Determine operating system
        self.os_name = platform.system()
        self.is_windows = self.os_name == 'Windows'
        self.is_mac = self.os_name == 'Darwin'
        self.is_linux = self.os_name == 'Linux'

        # Load window position and size
        config = load_config()
        if 'window_size' in config:
            self.master.geometry(config['window_size'])

        # Load previously selected folder from config
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

        # Scale factor for zooming
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0

        # Variables to handle panning
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0

        # Left pane - Listbox for image list with detection status
        self.listbox_frame = tk.Frame(master)
        self.listbox_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False)

        self.listbox = tk.Listbox(self.listbox_frame, width=50, height=20)
        self.listbox.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_image_select)

        # Ensure the listbox can scroll with the mouse wheel
        self.listbox.bind("<MouseWheel>", lambda event: self.on_listbox_scroll(event))
        self.listbox.bind("<Button-4>", lambda event: self.on_listbox_scroll(event))  # Linux scroll up
        self.listbox.bind("<Button-5>", lambda event: self.on_listbox_scroll(event))  # Linux scroll down

        # Add scrollbar to listbox
        self.scrollbar = tk.Scrollbar(self.listbox_frame, orient=tk.VERTICAL)
        self.scrollbar.config(command=self.listbox.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=self.scrollbar.set)

        # Right pane - Canvas for displaying images
        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Enter>", lambda event: self.canvas.focus_set())
        self.canvas.bind("<Leave>", lambda event: self.canvas.focus_set())
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows and macOS
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux scroll down

        # Navigation buttons frame
        nav_frame = tk.Frame(master)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.prev_button = tk.Button(nav_frame, text="Previous", command=self.show_previous_image)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.next_button = tk.Button(nav_frame, text="Next", command=self.show_next_image)
        self.next_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add status label and progress bar
        self.status_label = tk.Label(master, text="Ready")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress = ttk.Progressbar(master, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(side=tk.BOTTOM, fill=tk.X)

        # Populate listbox with file names and dates
        self.populate_listbox()

        # Start background processing for pupil detection
        threading.Thread(target=self.process_images_in_background, daemon=True).start()

        # Bind arrow keys for navigation
        self.master.bind('<Left>', lambda event: self.show_previous_image())
        self.master.bind('<Right>', lambda event: self.show_next_image())
        self.master.bind('<Up>', lambda event: self.show_previous_image())
        self.master.bind('<Down>', lambda event: self.show_next_image())

        # Save window size and position on close
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    # Handle listbox scrolling
    def on_listbox_scroll(self, event):
        if event.delta:
            self.listbox.yview_scroll(int(-1*(event.delta/120)), "units")
        elif event.num == 4:
            self.listbox.yview_scroll(-1, "units")
        elif event.num == 5:
            self.listbox.yview_scroll(1, "units")

    # Thread-safe method to update status label
    def update_status(self, message):
        self.status_label.after(0, lambda: self.status_label.config(text=message))

    # Thread-safe method to update progress bar
    def update_progress(self, value):
        self.progress.after(0, lambda: self.progress.config(value=value))

    # Populate the listbox with file names and dates
    def populate_listbox(self):
        for idx, (filename, _, datetime_taken) in enumerate(self.images):
            # Format date/time for display
            date_str = datetime_taken.strftime('%Y-%m-%d %H:%M:%S')
            if filename in self.eye_positions:
                self.listbox.insert(tk.END, f"{filename} ({date_str}) ✅")
            else:
                self.listbox.insert(tk.END, f"{filename} ({date_str}) ⏳")

    # Background thread to process images for pupil detection
    def process_images_in_background(self):
        total_images = len(self.images)
        self.progress['maximum'] = total_images
        for idx, (filename, image, datetime_taken) in enumerate(self.images):
            self.update_status(f"Processing image {idx + 1}/{total_images}: {filename}")
            self.update_progress(idx + 1)
            if filename not in self.eye_positions:
                left_pupil, right_pupil = detect_pupils_with_preprocessing(image)
                if left_pupil != (0, 0) and right_pupil != (0, 0):
                    save_eye_positions(filename, left_pupil, right_pupil, datetime_taken)
                    self.eye_positions[filename] = {
                        'pupils': [left_pupil, right_pupil],
                        'datetime_taken': datetime_taken
                    }
                    self.update_listbox_item(idx, filename, True)
                else:
                    self.update_listbox_item(idx, filename, False)
        self.update_status("Processing complete.")
        self.update_progress(0)

    # Update a specific item in the listbox (success or failure)
    def update_listbox_item(self, index, filename, success):
        datetime_taken = self.images[index][2]
        date_str = datetime_taken.strftime('%Y-%m-%d %H:%M:%S')
        status_icon = "✅" if success else "❌"
        new_text = f"{filename} ({date_str}) {status_icon}"
        self.listbox.delete(index)
        self.listbox.insert(index, new_text)

    # Load and display image at a specific index
    def show_image_at_index(self, index):
        if index < 0 or index >= len(self.images):
            return
        self.current_image_index = index
        filename, image, _ = self.images[index]
        self.current_image = image  # Store original image
        self.zoom_scale = 1.0  # Reset zoom scale when a new image is selected
        self.pan_x = 0
        self.pan_y = 0
        self.clicks = []  # Reset clicks when a new image is selected
        self.first_click_marker = None  # Reset first click marker
        pupil_data = self.eye_positions.get(filename, None)
        pupil_positions = pupil_data['pupils'] if pupil_data else None

        self.display_image(image, pupil_positions)
        # Update listbox selection
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self.listbox.activate(index)
        self.listbox.see(index)

    # Show previous image
    def show_previous_image(self):
        if self.current_image_index is not None and self.current_image_index > 0:
            self.show_image_at_index(self.current_image_index - 1)

    # Show next image
    def show_next_image(self):
        if self.current_image_index is not None and self.current_image_index < len(self.images) - 1:
            self.show_image_at_index(self.current_image_index + 1)

    # Load and display selected image on the right pane
    def on_image_select(self, event):
        selection = event.widget.curselection()
        if not selection:
            return
        index = selection[0]
        self.show_image_at_index(index)

    # Display image on canvas and draw pupils
    def display_image(self, image, pupil_positions):
        self.canvas.delete("all")

        # Resize image according to zoom scale
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        zoomed_width = int(pil_image.width * self.zoom_scale)
        zoomed_height = int(pil_image.height * self.zoom_scale)

        self.display_image_width = zoomed_width
        self.display_image_height = zoomed_height

        pil_image = pil_image.resize((zoomed_width, zoomed_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_image)

        # Calculate offsets for panning
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Center the image initially
        self.offset_x = (canvas_width - zoomed_width) // 2 + self.pan_x
        self.offset_y = (canvas_height - zoomed_height) // 2 + self.pan_y

        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)

        # Update scaling factors
        self.scale_factor_w = (zoomed_width / image.shape[1])
        self.scale_factor_h = (zoomed_height / image.shape[0])

        # Draw pupils
        if pupil_positions:
            left_pupil, right_pupil = pupil_positions

            # Map pupil positions to canvas coordinates
            left_pupil_canvas_x = left_pupil[0] * self.scale_factor_w + self.offset_x
            left_pupil_canvas_y = left_pupil[1] * self.scale_factor_h + self.offset_y
            right_pupil_canvas_x = right_pupil[0] * self.scale_factor_w + self.offset_x
            right_pupil_canvas_y = right_pupil[1] * self.scale_factor_h + self.offset_y

            # Draw circles on the canvas
            self.canvas.create_oval(
                left_pupil_canvas_x - 2, left_pupil_canvas_y - 2,
                left_pupil_canvas_x + 2, left_pupil_canvas_y + 2,
                outline="yellow", width=2
            )
            self.canvas.create_oval(
                right_pupil_canvas_x - 2, right_pupil_canvas_y - 2,
                right_pupil_canvas_x + 2, right_pupil_canvas_y + 2,
                outline="yellow", width=2
            )

        # If first click marker exists, redraw it
        if self.clicks:
            first_click = self.clicks[0]
            first_click_canvas_x = first_click[0] * self.scale_factor_w + self.offset_x
            first_click_canvas_y = first_click[1] * self.scale_factor_h + self.offset_y
            self.canvas.create_oval(
                first_click_canvas_x - 2, first_click_canvas_y - 2,
                first_click_canvas_x + 2, first_click_canvas_y + 2,
                outline="red", width=2
            )

    # Handle window resize event to adjust image display
    def on_canvas_resize(self, event):
        if self.current_image_index is not None:
            filename, image, _ = self.images[self.current_image_index]
            pupil_data = self.eye_positions.get(filename, None)
            pupil_positions = pupil_data['pupils'] if pupil_data else None
            self.display_image(self.current_image, pupil_positions)

    # Handle mouse wheel events for zooming
    def on_mouse_wheel(self, event):
        if self.current_image_index is None:
            return

        # Get mouse position relative to canvas
        mouse_x = self.canvas.canvasx(event.x)
        mouse_y = self.canvas.canvasy(event.y)

        # Windows and macOS
        if event.delta:
            delta = event.delta
            if self.is_windows or self.is_mac:
                delta = event.delta / 120  # Normalize delta to ±1 per scroll notch
        # Linux (event.num is 4 or 5)
        elif hasattr(event, 'num'):
            if event.num == 4:
                delta = 1
            elif event.num == 5:
                delta = -1
            else:
                delta = 0
        else:
            delta = 0

        # Calculate zoom factor
        if delta > 0:
            zoom_factor = 1.1 ** delta
        elif delta < 0:
            zoom_factor = 1 / (1.1 ** abs(delta))
        else:
            zoom_factor = 1

        # Limit zoom scale
        new_zoom_scale = self.zoom_scale * zoom_factor
        if new_zoom_scale < 0.1 or new_zoom_scale > 10:
            return
        self.zoom_scale = new_zoom_scale

        # Adjust pan to keep the image centered around the mouse pointer
        self.pan_x = (self.pan_x - mouse_x) * zoom_factor + mouse_x
        self.pan_y = (self.pan_y - mouse_y) * zoom_factor + mouse_y

        filename, image, _ = self.images[self.current_image_index]
        pupil_data = self.eye_positions.get(filename, None)
        pupil_positions = pupil_data['pupils'] if pupil_data else None
        self.display_image(self.current_image, pupil_positions)

    # Handle mouse button press for panning and pupil selection
    def on_button_press(self, event):
        if self.current_image_index is None:
            return

        self.start_x = event.x
        self.start_y = event.y

        # Check if shift key is pressed for selection
        if event.state & 0x0001:  # Shift key
            self.is_panning = False
            self.handle_pupil_selection(event)
        else:
            self.is_panning = True
            self.pan_start_x = event.x
            self.pan_start_y = event.y

    # Handle mouse dragging for panning
    def on_mouse_drag(self, event):
        if self.current_image_index is None or not self.is_panning:
            return

        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y

        self.pan_x += dx
        self.pan_y += dy

        self.pan_start_x = event.x
        self.pan_start_y = event.y

        filename, image, _ = self.images[self.current_image_index]
        pupil_data = self.eye_positions.get(filename, None)
        pupil_positions = pupil_data['pupils'] if pupil_data else None
        self.display_image(self.current_image, pupil_positions)

    # Handle mouse button release
    def on_button_release(self, event):
        if self.current_image_index is None:
            return
        if self.is_panning:
            self.is_panning = False

    # Handle pupil selection when shift key is pressed
    def handle_pupil_selection(self, event):
        # Calculate coordinates relative to the displayed image
        clicked_x = event.x
        clicked_y = event.y

        image_x = (clicked_x - self.offset_x) / self.scale_factor_w
        image_y = (clicked_y - self.offset_y) / self.scale_factor_h

        if 0 <= image_x <= self.current_image.shape[1] and 0 <= image_y <= self.current_image.shape[0]:
            self.clicks.append((int(image_x), int(image_y)))

            if len(self.clicks) == 1:
                # Draw a visual prompt at the first click location
                filename = self.images[self.current_image_index][0]
                pupil_data = self.eye_positions.get(filename, None)
                pupil_positions = pupil_data['pupils'] if pupil_data else None
                self.display_image(self.current_image, pupil_positions)
            elif len(self.clicks) == 2:
                left_pupil, right_pupil = self.clicks
                filename = self.images[self.current_image_index][0]
                datetime_taken = self.images[self.current_image_index][2]
                save_eye_positions(filename, left_pupil, right_pupil, datetime_taken)
                self.eye_positions[filename] = {
                    'pupils': [left_pupil, right_pupil],
                    'datetime_taken': datetime_taken
                }
                pupil_positions = [left_pupil, right_pupil]
                self.display_image(self.current_image, pupil_positions)
                self.update_listbox_item(self.current_image_index, filename, True)
                self.clicks = []
                self.first_click_marker = None  # Remove the first click marker

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
    init_db()
    root = tk.Tk()
    app = EyeReviewApp(root)
    root.mainloop()
