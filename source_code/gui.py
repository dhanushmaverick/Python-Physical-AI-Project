from source_code.main import *
import tkinter as tk
from tkinter import ttk
import cv2

# =====================================================
# IMAGE SETUP
# =====================================================
cv2.imwrite(
    "source_code/vision/object_segmentation/Img.png",
    cv2.resize(
        cv2.imread("source_code/vision/object_segmentation/Img.jpeg"),
        (650, 650)
    )
)

cv2.imwrite(
    "source_code/utility/img_pg1.png",
    cv2.resize(
        cv2.imread("source_code/utility/img.png"),
        (150, 150)
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
# =====================================================
# ROOT
# =====================================================
root = tk.Tk()
root.title("Physical AI Simulator")
root.geometry("1920x1080")
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

page1 = tk.Frame(container, bg=BG)
page1.grid(row=0, column=0, sticky="nsew")

page2 = tk.Frame(container, bg=BG)
page2.grid(row=0, column=0, sticky="nsew")

End_page = tk.Frame(container, bg=BG)
End_page.grid(row=0, column=0, sticky="nsew")

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

<<<<<<< HEAD
container = tk.Frame(root)
container.pack(fill="both", expand=True)
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)
#Page Definitions
HomePage = tk.Frame(container, bg="#1e1e2e")
HomePage.grid(row=0, column=0, sticky="nsew")
page1 = tk.Frame(container, bg="#1e1e2e")
page1.grid(row=0, column=0, sticky="nsew")
page2 = tk.Frame(container, bg="#282a36")
page2.grid(row=0, column=0, sticky="nsew")
page3 = tk.Frame(container, bg="#282a36")
page3.grid(row=0, column=0, sticky="nsew")
End_page = tk.Frame(container, bg="#282a36")
End_page.grid(row=0, column=0, sticky="nsew")
current_page = HomePage;
page_list = [HomePage,page1,page2,page3];
back_btn = tk.Button(root,text="Back",command=back, font=("Segoe UI", 12, "bold")).pack(side="left", padx=10, pady=10)
#HomePage
#back_btn = tk.Button(root,text="Back",command=back, font=("Segoe UI", 12, "bold")).pack(side="left", padx=10, pady=10)
cover_image = tk.PhotoImage(file = "source_code/utility/img.png")

title1 = tk.Label(
    HomePage,
    text="Home Page",
    font=("Segoe UI", 28, "bold"),
    bg="#1e1e2e",
    fg="white"
)
title1.pack(pady=40)
start_button = tk.Button(HomePage,text="Start",command=lambda: show_page(page1), font=("Segoe UI", 12, "bold"))
start_button.pack(pady=20)
exit_button_home = tk.Button(root,text="Exit",command=exit_app, font=("Segoe UI", 12, "bold"))
exit_button_home.pack(side="bottom", pady=10)
=======
def simulation(entry):
    if entry.get().strip() == "":
        show_page(End_page)
    else:
        run_multiple_cmds(["7", "8"])
        query(input_entry.get())
        run_command("6")
        show_page(End_page)
>>>>>>> 4c263593b4ab5370b88bae0c7133d3b553b9c5de

# =====================================================
# IMAGES
# =====================================================
cover_image = tk.PhotoImage(file="source_code/utility/img.png")
cover_image_pg1 = tk.PhotoImage(file="source_code/utility/img_pg1.png")
img_used = tk.PhotoImage(file="source_code/vision/object_segmentation/Img.png")

<<<<<<< HEAD
#Page1: Abt asking camera


desc1 = tk.Label(
    page1,
    text="Is this a new camera or no?",
    font=("Segoe UI", 14),
    bg="#1e1e2e",
    fg="#cfcfe6"
)
desc1.pack(pady=10)
btn1 = tk.Button(page1, text="Yes and clear calibration images", command=lambda: {run_new_camera_setup(True),show_page(page3)})
btn1.pack( padx=5,pady = 20)

btn2 = tk.Button(page1, text="Yes but keep existing calibration images", command=lambda: {run_new_camera_setup(False),show_page(page3)})
btn2.pack(padx=5,pady=20)
btn2 = tk.Button(page1, text="No", command=lambda: show_page(page2))
btn2.pack(padx=5,pady=25)
#
#btn1 = tk.Button(
 #   page1,
  #  text="Go To Page 2",
   # font=("Segoe UI", 12, "bold"),
    #bg="#6c63ff",
   # fg="white",
   # activebackground="#5848e5",
   # activeforeground="white",
   # relief="flat",
   # padx=20,
   # pady=10,
   # command=lambda: show_page(page2)
#)
#btn1.pack(pady=30)


#Page2

title2 = tk.Label(
    page2,
    text="Second Page",
    font=("Segoe UI", 28, "bold"),
    bg="#282a36",
    fg="white"
)
title2.pack(pady=40)

desc2 = tk.Label(
    page2,
    text="Was the camera or workspace moved?",
    font=("Segoe UI", 14),
    bg="#282a36",
    fg="#dcdcdc"
)
desc2.pack(pady=10)

btn2_yes = tk.Button(
    page2,
    text="Yes",
    font=("Segoe UI", 12, "bold"),
    bg="#ff6584",
    fg="white",
    activebackground="#e14d6c",
    activeforeground="white",
    relief="flat",
    padx=20,
    pady=10,
    command=lambda: {run_workspace_moved_setup(),show_page(page3)}
)
btn2_yes.pack(pady=30)
btn2_no = tk.Button(
    page2,
    text="No",
    font=("Segoe UI", 12, "bold"),
    bg="#ff6584",
    fg="white",
    activebackground="#e14d6c",
    activeforeground="white",
    relief="flat",
    padx=20,
    pady=10,
    command=lambda: show_page(page3)
)
btn2_no.pack(pady=30)
#Page3: Simulation and AI prompt
title3 = tk.Label(
    page3,
    text="What do you want to do now?",
    font=("Arial", 16, "bold"),
    fg="white",
    bg="#0f172a"
)
title3.pack(pady=40)
btn_op1 = tk.Button(
    page3,
    text="1. Find objects first\nThen prompt the task",
    font=("Arial", 12),
    bg="#1f2937",
    fg="white",
    width=30,
    height=3,
    command=lambda:{ print("\nOkay. I will find the objects first."),run_multiple_cmds(["7", "8"]),print("\nNow I will start the simulation."),run_command("6"),show_page(End_page)},
    relief="flat",
    activebackground="#374151"
)
btn_op1.pack(pady=10)
btn_op2 = tk.Button(
    page3,
    text="2. Directly prompt the task",
    font=("Arial", 12),
    bg="#2563eb",
    fg="white",
    width=30,
    height=3,
    command=lambda:{print("\nOkay. I will directly start the simulation."),run_command("6"),show_page(End_page)},
    relief="flat",
    activebackground="#1d4ed8"
)
btn_op2.pack(pady=10)
title = tk.Label(
=======
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
    font=("Segoe UI", 26, "bold"),
    bg=RED,
    fg="white",
    bd=0
).place(relx=0.5, rely=0.95, anchor="s")

tk.Button(
>>>>>>> 4c263593b4ab5370b88bae0c7133d3b553b9c5de
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
home_frame.place(relx=0.5, rely=0.5, anchor="center")

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
left.place(relx=0.15, rely=0.55, anchor="center")

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
    command=lambda: run_workspace_moved_setup(),
    font=("Segoe UI", 25, "bold"),
    bg=BLUE,
    fg=TEXT,
    bd=0
).pack(pady=15)

# =====================================================
# CENTER PANEL
# =====================================================
center = tk.Frame(page1, bg=BG)
center.place(relx=0.52, rely=0.5, anchor="center")
tk.Label(
    center,
    text="Camera and Workspace Setup",
    font=("Segoe UI", 36, "bold"),
    fg=TEXT,
    bg=BG
).pack(pady=20)

tk.Label(
    center,
    image=img_used,
    bg=BG
).pack()

tk.Label(
    center,
    text="User Query",
    font=("Segoe UI", 26, "bold"),
    fg=TEXT,
    bg=BG
).pack(pady=20)

input_entry = tk.Entry(
    center,
    font=("Segoe UI", 26),
    bg=GRAY,
    fg=TEXT,
    
    bd=0,width = 40,
    justify="center"
)
input_entry.pack(ipady=16)   # BIGGER INPUT HEIGHT

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
right.place(relx=0.85, rely=0.55, anchor="center")

tk.Label(
    right,
    text="Was the camera changed?",
    font=("Segoe UI", 28, "bold"),
    fg=MUTED,
    bg=BG
).pack(pady=10)

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
p2.place(relx=0.5, rely=0.5, anchor="center")

tk.Label(
    p2,
    text="Is this a new Camera?",
    font=("Segoe UI", 32, "bold"),
    fg=TEXT,
    bg=BG
).pack(pady=40)

tk.Button(
    p2,
    text="YES and clear calibration images",
    command=lambda: {run_new_camera_setup(True), show_page(page1)},
    font=("Segoe UI", 25, "bold"),
    bg=CARD,
    fg="white",
    bd=0
).pack(pady=10)

tk.Button(
    p2,
    text="YES but keep existing calibration images",
    command=lambda: {run_new_camera_setup(False), show_page(page1)},
    font=("Segoe UI", 25, "bold"),
    bg=BLACK,
    fg=TEXT,
    bd=0
).pack(pady=10)

# =====================================================
# END PAGE
# =====================================================
end_frame = tk.Frame(End_page, bg=BG)
end_frame.place(relx=0.5, rely=0.5, anchor="center")

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
    font=("Segoe UI", 26, "bold"),
    bg=GREEN,
    fg="black",
    bd=0
).pack(pady=20)

# =====================================================
# START
# =====================================================
show_page(HomePage)
root.mainloop()