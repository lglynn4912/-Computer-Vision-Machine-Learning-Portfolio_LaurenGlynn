from skimage import color, feature
import tkinter as tk


class ImageClicker:
    def __init__(self, image_path, n_points=4):
        self.root = tk.Tk()
        self.root.title("Image Clicker")

        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack()

        self.image = tk.PhotoImage(file=image_path)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        self.points = []
        self.n_points = n_points

        self.canvas.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red")

        if len(self.points) == self.n_points:
            self.root.destroy()

    def get_points(self):
        return self.points

    def run(self):
        self.root.mainloop()

#
def genSIFTMatches(img_s, img_d):
    # Convert images to grayscale
    gray_s = color.rgb2gray(img_s)
    gray_d = color.rgb2gray(img_d)

    # Compute SIFT features
    sift = feature.SIFT()
    sift.detect_and_extract(gray_s)
    Fs, Ds = sift.keypoints, sift.descriptors
    sift.detect_and_extract(gray_d)
    Fd, Dd = sift.keypoints, sift.descriptors

    # Match descriptors
    matches = feature.match_descriptors(Ds, Dd, cross_check=True)

    # Extract the locations of matched keypoints
    xs = Fs[matches[:, 0]]
    xd = Fd[matches[:, 1]]

    return xs, xd

