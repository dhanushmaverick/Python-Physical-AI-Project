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
        (300, 300)
    )
)


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


# =====================================================
# ROOT
# =====================================================
root = tk.Tk()
root.title("Physical AI Simulator")
root.geometry("1300x850")
root.configure(bg="#47b5e9")
#root.attributes("-fullscreen", True)
root.state('zoomed')  # Start maximized

container = tk.Frame(root, bg="#1e1e2e")
container.pack(fill="both", expand=True)

container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)


# =====================================================
# PAGES
# =====================================================
HomePage = tk.Frame(container, bg="#47b5e9")
HomePage.grid(row=0, column=0, sticky="nsew")

page1 = tk.Frame(container, bg="#47b5e9")
page1.grid(row=0, column=0, sticky="nsew")

page2 = tk.Frame(container, bg="#47b5e9")
page2.grid(row=0, column=0, sticky="nsew")

page3 = tk.Frame(container, bg="#47b5e9")
page3.grid(row=0, column=0, sticky="nsew")

End_page = tk.Frame(container, bg="#47b5e9")
End_page.grid(row=0, column=0, sticky="nsew")

current_page = HomePage
page_list = [HomePage, page1, page2, page3]


# =====================================================
# NAV BUTTONS (ONLY SIZE CHANGED)
# =====================================================
tk.Button(
    root,
    text="Back",
    command=back,
    font=("Segoe UI", 18, "bold"),   # bigger
    bg="#374151",
    fg="white",
    padx=35,                        # bigger
    pady=18                        # bigger
).pack(side="left", padx=10)

tk.Button(
    root,
    text="Exit",
    command=exit_app,
    font=("Segoe UI", 18, "bold"),   # bigger
    bg="#ef4444",
    fg="white",
    padx=35,
    pady=18
).pack(side="bottom", pady=10)


# =====================================================
# IMAGES
# =====================================================
cover_image = tk.PhotoImage(file="source_code/utility/img.png")
img_used = tk.PhotoImage(file="source_code/vision/object_segmentation/Img.png")


# =====================================================
# CENTERING HELPER
# =====================================================
def center_frame(parent):
    frame = tk.Frame(parent, bg=parent["bg"])
    frame.place(relx=0.5, rely=0.5, anchor="center")
    return frame


# =====================================================
# HOME PAGE
# =====================================================
home_frame = center_frame(HomePage)

tk.Label(
    home_frame,
    text="Welcome to Physical AI enabled\n Vision Pick and Place Simulator",
    font=("Segoe UI", 34, "bold"),
    bg="#1e1e2e",
    fg="white",
    justify="center"
).pack(pady=30)


tk.Label(
    HomePage,
    image=cover_image,
    bg="#1e1e2e"
).place(relx=0.95, rely=0.05, anchor="ne")
tk.Label(
    page1,
    image=cover_image,
    bg="#1e1e2e"
).place(relx=0.95, rely=0.05, anchor="ne")

tk.Button(
    home_frame,
    text="START",
    command=lambda: show_page(page1),
    font=("Segoe UI", 26, "bold"),   # bigger ONLY
    bg="#22c55e",
    fg="white",
    padx=80,                        # bigger ONLY
    pady=30                        # bigger ONLY
).pack(pady=40)


# =====================================================
# PAGE 1
# =====================================================
page1_frame = center_frame(page1)


tk.Label(
    page1_frame,
    text="Camera and Workspace Setup",
    font=("Segoe UI", 28, "bold"),
    bg="#282a36",
    fg="white"
).pack(pady=20)


tk.Label(
    page1,
    image=img_used,
    bg="#282a36"
).place(relx=0.05, rely=0.1, anchor="nw")


tk.Label(
    page1_frame,
    text="Was the camera or workspace moved?",
    font=("Segoe UI", 20),
    bg="#282a36",
    fg="#dcdcdc"
).pack(pady=20)


tk.Button(
    page1_frame,
    text="YES",
    font=("Segoe UI", 22, "bold"),   # bigger only
    bg="#ff6584",
    fg="white",
    padx=80,
    pady=30,
    command=lambda: {run_workspace_moved_setup(), show_page(page3)}
).pack(pady=10)


tk.Button(
    page1_frame,
    text="NO",
    font=("Segoe UI", 22, "bold"),
    bg="#ff6584",
    fg="white",
    padx=80,
    pady=30,
    command=lambda: show_page(page3)
).pack(pady=10)


tk.Button(
    page1_frame,
    text="CALIBRATE CAMERA",
    font=("Segoe UI", 22, "bold"),
    bg="#2563eb",
    fg="white",
    padx=80,
    pady=30,
    command=lambda: show_page(page2)
).pack(pady=20)


tk.Label(
    page1_frame,
    text="Enter the order of blocks to be stacked",
    font=("Segoe UI", 18),
    bg="#282a36",
    fg="#dcdcdc"
).pack(pady=10)


input_entry = tk.Entry(
    page1_frame,
    font=("Segoe UI", 18),
    bg="#1e1e2e",
    fg="white",
    width=30
)
input_entry.pack(ipady=10, pady=10)


tk.Button(
    page1_frame,
    text="SUBMIT",
    font=("Segoe UI", 22, "bold"),
    bg="#22c55e",
    fg="white",
    padx=80,
    pady=30,
    command=lambda: {
        print(input_entry.get()),
        show_page(page3)
    }
).pack(pady=20)


# =====================================================
# PAGE 2 (TEXT UNCHANGED)
# =====================================================
page2_frame = center_frame(page2)

tk.Label(
    page2_frame,
    text="Is this a new camera or no?",
    font=("Segoe UI", 22, "bold"),
    bg="#1e1e2e",
    fg="white"
).pack(pady=40)


tk.Button(
    page2_frame,
    text="YES and clear calibration images",
    font=("Segoe UI", 18, "bold"),
    bg="#ff6584",
    fg="white",
    padx=60,
    pady=25,
    command=lambda: {run_new_camera_setup(True), show_page(page3)}
).pack(pady=15)


tk.Button(
    page2_frame,
    text="YES but keep existing calibration images",
    font=("Segoe UI", 18, "bold"),
    bg="#ff6584",
    fg="white",
    padx=60,
    pady=25,
    command=lambda: {run_new_camera_setup(False), show_page(page3)}
).pack(pady=15)


tk.Button(
    page2_frame,
    text="No",
    font=("Segoe UI", 18, "bold"),
    bg="#ff6584",
    fg="white",
    padx=60,
    pady=25,
    command=lambda: show_page(page3)
).pack(pady=15)


# =====================================================
# PAGE 3 (UNCHANGED TEXT)
# =====================================================
page3_frame = center_frame(page3)

tk.Label(
    page3_frame,
    text="What do you want to do now?",
    font=("Segoe UI", 24, "bold"),
    fg="white",
    bg="#282a36"
).pack(pady=40)


tk.Button(
    page3_frame,
    text="Find objects first,\n then Prompt the task?",
    font=("Segoe UI", 18, "bold"),
    bg="#1f2937",
    fg="white",
    width=35,
    height=4,
    command=lambda: {
        run_multiple_cmds(["7", "8"]),
        query(input_entry.get()),
        run_command("6"),
        show_page(End_page)
    }
).pack(pady=20)


tk.Button(
    page3_frame,
    text="Directly Prompt the task?",
    font=("Segoe UI", 18, "bold"),
    bg="#2563eb",
    fg="white",
    width=35,
    height=4,
    command=lambda: {
        query(input_entry.get()),
        run_command("6"),
        show_page(End_page)
    }
).pack(pady=20)


# =====================================================
# END PAGE (UNCHANGED)
# =====================================================
end_frame = center_frame(End_page)

tk.Label(
    end_frame,
    text="Simulation Complete!",
    font=("Segoe UI", 28, "bold"),
    fg="#22c55e",
    bg="#282a36"
).pack(pady=30)


tk.Button(
    end_frame,
    text="RESTART",
    font=("Segoe UI", 18, "bold"),
    bg="#2563eb",
    fg="white",
    padx=50,
    pady=20,
    command=lambda: show_page(HomePage)
).pack(pady=20)


# =====================================================
# START
# =====================================================
show_page(HomePage)
root.mainloop()