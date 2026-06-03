

from source_code.main import *
import tkinter as tk
from tkinter import ttk,messagebox
import cv2
from source_code.utility.paths import *

# =====================================================
# IMAGE SETUP
# =====================================================
cv2.imwrite(
    OBJ_SEGMENTATION_DIR/"Img.png",
     cv2.resize(
         cv2.imread(OBJ_SEGMENTATION_DIR/"Img.jpeg"),
         (500, 500)
     )
 )

cv2.imwrite(
    "source_code/utility/img_pg1.png",
    cv2.resize(
        cv2.imread("source_code/utility/img.png"),
        (150, 150)
    )
)
cv2.imwrite(
    "source_code/utility/img_home.png",
    cv2.resize(
        cv2.imread("source_code/utility/img.png"),
        (300, 300)
    )
)
# =====================================================
# THEME (TESLA AUTOPILOT STYLE HUD)
# =====================================================
BG = "#000000"
CARD = "#d80c5a"
CARD_2 = "#d80c5a"
BLACK = "#000000"
TEXT = "#e8eef2"
MUTED = "#93a1ad"

ACCENT = "#3ea6ff"
ACCENT_2 = "#7dd3fc"

RED = "#ff3b30"
GREEN = "#13ec0c"
BLUE = "#0b53ee"
GRAY = "#2b2f36"
DARK_GREEN = "#0c3b0a"
# =====================================================
# ROOT
# =====================================================
root = tk.Tk()
root.attributes("-fullscreen", "1")  # Start maximized
# =====================================================
# AUTO SCALING
# =====================================================
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()

BASE_WIDTH = 1920
BASE_HEIGHT = 1080
scale = min(screen_w / BASE_WIDTH, screen_h / BASE_HEIGHT)
root.tk.call("tk", "scaling", scale)

root.title("Physical AI Simulator")
root.geometry(f"{screen_w}x{screen_h}")

root.tk.call("tk", "scaling", scale)

root.configure(bg=BG)

container = tk.Frame(root, bg=BG)
container.pack(fill="both", expand=True)

container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)

# =====================================================
# PAGES
# =====================================================
HomePage = tk.Frame(container, bg=BG)
HomePage.grid(row=0, column=0, sticky="nsew")
HomePage.grid_rowconfigure(0, weight=1)
HomePage.grid_columnconfigure(0, weight=1)
page1 = tk.Frame(container, bg=BG)
page1.grid(row=0, column=0, sticky="nsew")
page1.grid_rowconfigure(0, weight=1)

page1.grid_columnconfigure(0, weight=1)
page1.grid_columnconfigure(1, weight=2)
page1.grid_columnconfigure(2, weight=1)
page2 = tk.Frame(container, bg=BG)
page2.grid(row=0, column=0, sticky="nsew")
page2.grid_rowconfigure(0, weight=1)
page2.grid_columnconfigure(0, weight=1)
End_page = tk.Frame(container, bg=BG)
End_page.grid(row=0, column=0, sticky="nsew")
End_page.grid_rowconfigure(0, weight=1)
End_page.grid_columnconfigure(0, weight=1)
current_page = HomePage
page_list = [HomePage, page1, page2, End_page]

def show_page(page):
    global current_page
    page.tkraise()
    current_page = page

def exit_app():
    root.destroy()

def back():
    i = page_list.index(current_page)
    if i > 0:
        show_page(page_list[i - 1])
    else:
        show_page(HomePage)

def simulation(entry):
    if entry.get().strip() == "":
        show_page(End_page)
    else:
        run_multiple_cmds(["7", "8"])
        file = open(AI_MOTION_PLAN_PATH, "w")
        file.write("")
        file.close()  # Clear previous motion plan
        query(input_entry.get())
        run_command("6")
        show_page(End_page)
def on_resize(event):
    scale = min(
        event.width / BASE_WIDTH,
        event.height / BASE_HEIGHT
    )

    new_scale = max(0.8, scale)

    root.tk.call("tk", "scaling", new_scale)

root.bind("<Configure>", on_resize)
def calibrate_click():
    messagebox.showinfo("Instructions", "1. Press s to save image for calibration.\n2. Press q to close the camera when done.\n3. Around 20 pictures of the workspace should be enough for a good calibration.\n\n\nRules to consider: \nRule 1: All four calibration corners must remain visible.\nRule 2: The calibration board may be translated or rotated.\nRule 3: One calibration corner should coincide with the desired world origin.\nWorld Origin = (0,0) \nRule 4: When selecting calibration points, begin at the origin and proceed anti-clockwise.\nRule 5: The usable workspace is restricted to the camera view from the robot home position.\n")

    return
def Retake_click():
    messagebox.showinfo("Instructions", "1. Press Enter key to retake image to be used for simulation \n2. Press q to close the camera.")
    return

# =====================================================
# IMAGES
# =====================================================
cover_image = tk.PhotoImage(file="source_code/utility/img_home.png")
cover_image_pg1 = tk.PhotoImage(file="source_code/utility/img_pg1.png")
img_used = tk.PhotoImage(file="source_code/vision/object_segmentation/Img.png")

# =====================================================
# NAV BUTTONS
# =====================================================
tk.Button(
    page1,
    text="Back",
    command=back,
    font=("Segoe UI", 22, "bold"),
    bg=BLACK,
    fg=TEXT,
    bd=0
).place(relx=0.02, rely=0.95, anchor="sw")
tk.Button(
    page2,
    text="Back",
    command=back,
    font=("Segoe UI", 22, "bold"),
    bg=BLACK,
    fg=TEXT,
    bd=0
).place(relx=0.02, rely=0.95, anchor="sw")
tk.Button(
    HomePage,
    text="Exit",
    command=exit_app,
    font=("Segoe UI", 36, "bold"),
    bg=RED,
    fg="white",
    bd=0
).place(relx=0.5, rely=0.95, anchor="s")

tk.Button(
    End_page,
    text="Exit",
    command=exit_app,
    font=("Segoe UI", 26, "bold"),
    bg=RED,
    fg="white",
    bd=0
).place(relx=0.5, rely=0.95, anchor="s")

# =====================================================
# HOME PAGE
# =====================================================
home_frame = tk.Frame(HomePage, bg=BG)
home_frame.grid(row=0, column=0)

tk.Label(
    home_frame,
    text="Welcome to Physical AI enabled\n Vision Pick and Place Simulator",
    font=("Segoe UI", 36, "bold"),
    fg=TEXT,
    bg=BG,
    justify="center"
).pack(pady=30)

tk.Label(
    HomePage,
    image=cover_image,
    bg=BG
).place(relx=0.95, rely=0.05, anchor="ne")

tk.Button(
    home_frame,
    text="START",
    command=lambda: show_page(page1),
    font=("Segoe UI", 30, "bold"),
    bg=GREEN,
    fg="black",
    bd=0
).pack(pady=40)

# =====================================================
# PAGE 1 TITLE
# =====================================================

# =====================================================
# LEFT PANEL
# =====================================================
left = tk.Frame(page1, bg=BG)
left.grid(row=0, column=0, sticky="", padx=20)

tk.Label(
    left,
    text="Was the workspace changed?",
    font=("Segoe UI", 28, "bold"),
    fg=MUTED,
    bg=BG
).pack(pady=10)

tk.Button(
    left,
    text="Calibrate Workspace",
    command=lambda: {calibrate_click(), run_workspace_moved_setup()},
    font=("Segoe UI", 25, "bold"),
    bg=BLUE,
    fg=TEXT,
    bd=0
).pack(pady=15,expand=True)

# =====================================================
# CENTER PANEL
# =====================================================
center = tk.Frame(page1, bg=BG)
center.grid(row=0, column=1, sticky="", padx=20)
tk.Label(
    center,
    text="Camera and Workspace Setup",
    font=("Segoe UI", 36, "bold"),
    fg=TEXT,
    bg=BG
).pack(pady=20)

img_label=tk.Label(
    center,
    image=img_used,
    bg=BG
)
img_label.pack()
def update_img():
    img = tk.PhotoImage(file=OBJ_SEGMENTATION_DIR/"Img.png")
    img_label.config(image=img)
    img_label.image=img
tk.Button(
    center,
    text="Retake Image",
    command= lambda:{Retake_click(),run_command("2"),update_img()},
    font=("Segoe UI", 25, "bold"),
    bg=DARK_GREEN ,
    fg=TEXT,
    bd=0
).pack(pady=15)
input_entry = tk.Entry(
    center,
    font=("Segoe UI", 26),
    bg=GRAY,
    fg=TEXT,
    
    bd=0,width = 40,
    justify="center"
)
input_entry.pack(ipady=16)   # BIGGER INPUT HEIGHT
input_entry.insert(0,"User Query")

def on_focus_in(event):
    if input_entry.get() == "User Query":
        input_entry.delete(0, tk.END)
        input_entry.config(fg="#cccccc75")

def on_focus_out(event):
    if not input_entry.get():
        input_entry.insert(0, "User Query")
        input_entry.config(fg="cccccc75")

input_entry.bind("<FocusIn>", on_focus_in)
input_entry.bind("<FocusOut>", on_focus_out)



bottom_buttons = tk.Frame(center, bg=BG)
bottom_buttons.pack(pady=20)

tk.Button(
    bottom_buttons,
    text="Simulate",
    command=lambda: simulation(input_entry),
    font=("Segoe UI", 25, "bold"),
    bg=GREEN,
    fg="black",
    bd=0
).pack(side="left", padx=10)

tk.Button(
    bottom_buttons,
    text="Exit",
    command=exit_app,
    font=("Segoe UI", 25, "bold"),
    bg=RED,
    fg="white",
    bd=0
).pack(side="left", padx=10)

# =====================================================
# RIGHT PANEL
# =====================================================
right = tk.Frame(page1, bg=BG)
right.grid(row=0, column=2, sticky="", padx=20)

tk.Label(
    right,
    text="Want to recalibrate your camera?",
    font=("Segoe UI", 28, "bold"),
    fg=MUTED,
    bg=BG
).pack(pady=10,expand=True)

tk.Button(
    right,
    text="Calibrate Camera",
    command=lambda: show_page(page2),
    font=("Segoe UI", 25, "bold"),
    bg=CARD,
    fg=TEXT,
    bd=0
).pack(pady=15)

# =====================================================
# PAGE 2
# =====================================================
p2 = tk.Frame(page2, bg=BG)
p2.grid(row=0, column=0)

tk.Label(
    p2,
    text="Do you want to recalibrate your camera from Scratch?",
    font=("Segoe UI", 32, "bold"),
    fg=TEXT,
    bg=BG
).pack(pady=40)

tk.Button(
    p2,
    text="Yes, clear all calibration images",
    command=lambda: {calibrate_click(), run_new_camera_setup(True), show_page(page1)},
    font=("Segoe UI", 25, "bold"),
    bg=CARD,
    fg="white",
    bd=0
).pack(pady=10)

tk.Button(
    p2,
    text="No, take more images to improve calibration",
    command=lambda: {calibrate_click(), run_new_camera_setup(False), show_page(page1)},
    font=("Segoe UI", 25, "bold"),
    bg=BLUE,
    fg=TEXT,
    bd=0
).pack(pady=10)

# =====================================================
# END PAGE
# =====================================================
end_frame = tk.Frame(End_page, bg=BG)
end_frame.grid(row=0, column=0)

tk.Label(
    end_frame,
    text="Simulation Complete!",
    font=("Segoe UI", 32, "bold"),
    fg=ACCENT,
    bg=BG
).pack(pady=30)

tk.Button(
    end_frame,
    text="RESTART",
    command=lambda: show_page(HomePage),
    font=("Segoe UI", 30, "bold"),
    bg=GREEN,
    fg="black",
    bd=0
).pack(pady=20)

# =====================================================
# START
# =====================================================
show_page(HomePage)
root.mainloop()