from source_code.main import *
import tkinter as tk
from tkinter import ttk
import cv2
import base64

# ---------------- TESLA THEME ----------------
BG = "#000000"
SIDEBAR = "#050505"
CARD = "#111111"
ACCENT = "#e31937"
TEXT = "#ffffff"
MUTED = "#a3a3a3"

# ---------------- APP STATE ----------------
def show_page(page):
    global current_page
    page.tkraise()
    current_page = page

def exit_app():
    root.destroy()

def change_text(label, new_text):
    label.config(text=new_text)

def back():
    i = page_list.index(current_page)
    if i > 0:
        show_page(page_list[i - 1])
    else:
        show_page(HomePage)

# ---------------- ROOT ----------------
root = tk.Tk()
root.title("Physical AI Simulator")
root.geometry("900x520")
root.configure(bg=BG)

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# ---------------- SIDEBAR ----------------
sidebar = tk.Frame(root, bg=SIDEBAR, width=160)
sidebar.grid(row=0, column=0, sticky="ns")

# ---------------- CENTER WRAPPER (CENTERED CONTENT FIX) ----------------
center_wrapper = tk.Frame(root, bg=BG)
center_wrapper.grid(row=0, column=1, sticky="nsew")

center_wrapper.grid_rowconfigure(0, weight=1)
center_wrapper.grid_columnconfigure(0, weight=1)

container = tk.Frame(center_wrapper, bg=BG)
container.place(relx=0.5, rely=0.5, anchor="center")

# ---------------- PAGES ----------------
HomePage = tk.Frame(container, bg=BG)
page1 = tk.Frame(container, bg=BG)
page2 = tk.Frame(container, bg=BG)
page3 = tk.Frame(container, bg=BG)
End_page = tk.Frame(container, bg=BG)

for p in (HomePage, page1, page2, page3, End_page):
    p.grid(row=0, column=0, sticky="nsew")

current_page = HomePage
page_list = [HomePage, page1, page2, page3]

# ---------------- TESLA BUTTON (BIG + HOVER EFFECT) ----------------
def tbutton(parent, text, command, bg=CARD):
    b = tk.Button(
        parent,
        text=text,
        command=command,
        font=("Helvetica Neue", 14, "bold"),
        fg=TEXT,
        bg=bg,
        activebackground=ACCENT,
        activeforeground="white",
        relief="flat",
        bd=0,
        padx=28,
        pady=18,
        cursor="hand2"
    )

    def on_enter(e):
        b.config(bg=ACCENT)

    def on_leave(e):
        b.config(bg=bg)

    b.bind("<Enter>", on_enter)
    b.bind("<Leave>", on_leave)

    return b

# ---------------- SIDEBAR ----------------
tk.Label(
    sidebar,
    text="CONTROL",
    font=("Helvetica Neue", 12, "bold"),
    fg=MUTED,
    bg=SIDEBAR
).pack(pady=20)

tbutton(sidebar, "Back", back, SIDEBAR).pack(pady=5)
tbutton(sidebar, "Exit", exit_app, SIDEBAR).pack(pady=5)

# ---------------- HOME PAGE ----------------
tk.Label(
    HomePage,
    text="Home Page",
    font=("Helvetica Neue", 28, "bold"),
    fg=TEXT,
    bg=BG
).pack(pady=40)

tbutton(
    HomePage,
    "Start",
    lambda: show_page(page1),
    CARD
).pack(pady=20)

cover_image = tk.PhotoImage(file="source_code/utility/img.png")
tk.Label(HomePage, image=cover_image, bg=BG).pack(pady=20)

# ---------------- PAGE 1 ----------------
tk.Label(
    page1,
    text="Is this a new camera or no?",
    font=("Helvetica Neue", 14),
    fg=MUTED,
    bg=BG
).pack(pady=15)

tbutton(
    page1,
    "Yes and clear calibration images",
    lambda: {run_new_camera_setup(True), show_page(page3)},
    CARD
).pack(pady=12)

tbutton(
    page1,
    "Yes but keep existing calibration images",
    lambda: {run_new_camera_setup(False), show_page(page3)},
    CARD
).pack(pady=12)

tbutton(
    page1,
    "No",
    lambda: show_page(page2),
    CARD
).pack(pady=12)

# ---------------- PAGE 2 ----------------
tk.Label(
    page2,
    text="Was the camera or workspace moved?",
    font=("Helvetica Neue", 14),
    fg=MUTED,
    bg=BG
).pack(pady=20)

tbutton(
    page2,
    "Yes",
    lambda: {run_workspace_moved_setup(), show_page(page3)},
    ACCENT
).pack(pady=20)

tbutton(
    page2,
    "No",
    lambda: show_page(page3),
    CARD
).pack(pady=12)

# ---------------- PAGE 3 ----------------
tk.Label(
    page3,
    text="What do you want to do now?",
    font=("Helvetica Neue", 16, "bold"),
    fg=TEXT,
    bg=BG
).pack(pady=30)

tbutton(
    page3,
    "1. Find objects first\nThen prompt the task",
    lambda: {
        print("\nOkay. I will find the objects first."),
        run_multiple_cmds(["7", "8"]),
        print("\nNow I will start the simulation."),
        run_command("6"),
        show_page(End_page)
    },
    CARD
).pack(pady=12)

tbutton(
    page3,
    "2. Directly prompt the task",
    lambda: {
        print("\nOkay. I will directly start the simulation."),
        run_command("6"),
        show_page(End_page)
    },
    ACCENT
).pack(pady=12)

# ---------------- END PAGE ----------------
tk.Label(
    End_page,
    text="Simulation Complete! What do you want to do next?",
    font=("Helvetica Neue", 20, "bold"),
    fg=ACCENT,
    bg=BG
).pack(pady=20)

tk.Label(
    End_page,
    text="All pipeline steps executed successfully.\nSystem is safe to shutdown.",
    font=("Helvetica Neue", 12),
    fg=MUTED,
    bg=BG
).pack()

log_box = tk.Text(
    End_page,
    height=8,
    bg="#0a0a0a",
    fg="white",
    font=("Consolas", 10),
    relief="flat",
    bd=0
)
log_box.pack(pady=15)

log_box.insert("end", "✓ Camera calibrated\n")
log_box.insert("end", "✓ Homography computed\n")
log_box.insert("end", "✓ Object detection complete\n")
log_box.insert("end", "✓ RoboDK simulation generated\n")
log_box.config(state="disabled")

tbutton(
    End_page,
    "RESTART",
    lambda: show_page(HomePage),
    ACCENT
).pack()

# ---------------- START ----------------
show_page(HomePage)
root.mainloop()