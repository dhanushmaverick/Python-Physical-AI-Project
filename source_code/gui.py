from source_code.main import *
import tkinter as tk
from tkinter import ttk
import cv2
import base64
cv2.imwrite("source_code/vision/object_segmentation/Img.png", cv2.resize(cv2.imread("source_code/vision/object_segmentation/Img.jpeg"),(300,300)))
def show_page(page):
    global current_page
    page.tkraise()
    current_page = page


def exit_app():
    root.destroy()
def change_text(label,new_text):
    label.config(text=new_text)
def back():
    i = page_list.index(current_page)
    if i>0:
        show_page(page_list[i-1])
    else: show_page(HomePage)
root = tk.Tk()
root.title("Physical AI Simulator")
root.geometry("700x450")
root.configure(bg="#1e1e2e")
#log_box = tk.Text(root, height=20, bg="#111827", fg="white", font=("Consolas", 11))
#log_box.pack(fill="both", expand=True, padx=10, pady=10)

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
img_used = tk.PhotoImage(file = "source_code/vision/object_segmentation/Img.png")
title1 = tk.Label(
    HomePage,
    text="Welcome to Physical AI enabled Vision Pick and Place Simulator",
    font=("Segoe UI", 28, "bold"),
    bg="#1e1e2e",
    fg="white"
)
title1.pack(pady=40)
start_button = tk.Button(HomePage,text="Start",command=lambda: show_page(page1), font=("Segoe UI", 12, "bold"))
start_button.pack(pady=20)
exit_button_home = tk.Button(root,text="Exit",command=exit_app, font=("Segoe UI", 12, "bold"))
exit_button_home.pack(side="bottom", pady=10)

label_img = tk.Label(HomePage, image = cover_image, bg="#1e1e2e")
label_img.pack(pady=20)

#Page1: Abt asking camera
title2 = tk.Label(
    page1,
    text="Camera and Workspace Setup",
    font=("Segoe UI", 28, "bold"),
    bg="#282a36",
    fg="white"
)
title2.pack(pady=40)
label_img2= tk.Label(page1, image = cover_image, bg="#1e1e2e")
label_img2.pack(side="right", pady=20)
label_imgUsed= tk.Label(page1, image = img_used, bg="#1e1e2e")
label_imgUsed.pack(side="left", pady=20)
desc2 = tk.Label(
    page1,
    text="Was the camera or workspace moved?",
    font=("Segoe UI", 14),
    bg="#282a36",
    fg="#dcdcdc"
)
desc2.pack(pady=10)

btn2_yes = tk.Button(
    page1,
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
    page1,
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
btn_calibrate = tk.Button(
    page1,
    text="Calibrate Camera",
    font=("Segoe UI", 12, "bold"),
    bg="#ff6584",
    fg="white",
    activebackground="#e14d6c",
    activeforeground="white",
    relief="flat",
    padx=20,
    pady=10,
    command=lambda: show_page(page2)
)
btn_calibrate.pack(pady=30)
Input_text = tk.Label(
    page1,
    text="Enter the order of blocks to be stacked",
    font=("Segoe UI", 14),
    bg="#282a36",
    fg="#dcdcdc"
)
Input_text.pack(pady=10)
input_entry = tk.Entry(
    page1,
    font=("Segoe UI", 12),
    bg="#1e1e2e",
    fg="white",
    relief="flat",
    width=30
)
input_entry.pack(pady=10)
btn_submit = tk.Button(
    page1,
    text="Submit",
    font=("Segoe UI", 12, "bold"),
    bg="#ff6584",
    fg="white",
    activebackground="#e14d6c",
    activeforeground="white",
    relief="flat",
    padx=20,
    pady=10,
    command=lambda: {print(f"Order submitted: {input_entry.get()}"),show_page(page3)}
)
btn_submit.pack(pady=10)
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


desc1 = tk.Label(
    page2,
    text="Is this a new camera or no?",
    font=("Segoe UI", 14),
    bg="#1e1e2e",
    fg="#cfcfe6"
)
desc1.pack(pady=10)
btn1 = tk.Button(page2, text="Yes and clear calibration images", command=lambda: {run_new_camera_setup(True),show_page(page3)})
btn1.pack( padx=5,pady = 20)

btn2 = tk.Button(page2, text="Yes but keep existing calibration images", command=lambda: {run_new_camera_setup(False),show_page(page3)})
btn2.pack(padx=5,pady=20)
btn2 = tk.Button(page2, text="No", command=lambda: show_page(page3))
btn2.pack(padx=5,pady=25)
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
    command=lambda:{ print("\nOkay. I will find the objects first."),run_multiple_cmds(["7", "8"]),query(input_entry.get()),print("\nNow I will start the simulation."),run_command("6"),show_page(End_page)},
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
    command=lambda:{query(input_entry.get()),print("\nOkay. I will directly start the simulation."),run_command("6"),show_page(End_page)},
    relief="flat",
    activebackground="#1d4ed8"
)
btn_op2.pack(pady=10)
title = tk.Label(
    End_page,
    text="Simulation Complete! What do you want to do next?",
    font=("Arial", 22, "bold"),
    fg="#22c55e",
    bg="#0f172a"
)
status_frame = tk.Frame(root, bg="#111827", padx=20, pady=20)
status_frame.pack(pady=10)

status_label = tk.Label(
    End_page,
    text="All pipeline steps executed successfully.\nSystem is safe to shutdown.",
    font=("Arial", 12),
    fg="white",
    bg="#111827",
    justify="center"
)
status_label.pack()
title.pack(pady=20)
log_box = tk.Text(
    End_page,
    height=8,
    bg="#0b1220",
    fg="#10b981",
    font=("Consolas", 10),
    relief="flat"
)
log_box.pack(fill="x", padx=20, pady=15)

log_box.insert("end", "✓ Camera calibrated\n")
log_box.insert("end", "✓ Homography computed\n")
log_box.insert("end", "✓ Object detection complete\n")
log_box.insert("end", "✓ RoboDK simulation generated\n")

log_box.config(state="disabled")
restart_btn = tk.Button(
    End_page,
    text="RESTART",
    command=lambda: show_page(HomePage),
    bg="#2563eb",
    fg="white",
    font=("Arial", 12),
    width=12
)
restart_btn.pack(side="left", padx=10)
show_page(HomePage)

root.mainloop()