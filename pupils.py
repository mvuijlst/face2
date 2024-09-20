import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

class PupilTaggerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pupil Tagger")

        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack()

        # Buttons to load images and save pupil positions
        self.load_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(self.root, text="Save Pupils", command=self.save_pupils)
        self.save_button.pack(side=tk.RIGHT)

        self.next_button = tk.Button(self.root, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT)

        self.prev_button = tk.Button(self.root, text="Previous", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT)

        self.image_list = []
        self.image_index = 0
        self.current_image = None
        self.pupil_positions = []

    def load_image(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.image_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            self.display_image()

    def display_image(self):
        self.canvas.delete("all")
        if self.image_list:
            self.current_image = Image.open(self.image_list[self.image_index])
            self.tk_image = ImageTk.PhotoImage(self.current_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self.canvas.bind("<Button-1>", self.mark_pupil)
            self.pupil_positions = []

    def mark_pupil(self, event):
        x, y = event.x, event.y
        self.pupil_positions.append((x, y))
        self.canvas.create_oval(x-5, y-5, x+5, y+5, outline="red", width=2)

    def save_pupils(self):
        if self.image_list:
            image_name = os.path.basename(self.image_list[self.image_index])
            with open(f"{image_name}_pupils.txt", "w") as f:
                for x, y in self.pupil_positions:
                    f.write(f"{x},{y}\n")

    def next_image(self):
        if self.image_list:
            self.image_index = (self.image_index + 1) % len(self.image_list)
            self.display_image()

    def prev_image(self):
        if self.image_list:
            self.image_index = (self.image_index - 1) % len(self.image_list)
            self.display_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = PupilTaggerApp(root)
    root.mainloop()
