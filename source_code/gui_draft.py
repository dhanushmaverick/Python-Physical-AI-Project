import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import threading
import time

# =========================
# COMMANDS (same as yours)
# =========================
RUN_COMMANDS = {
    "1": {
        "name": "Test camera",
        "cmd": [sys.executable, "-m", "source_code.vision.camera.test_camera"],
    },
    "2": {
        "name": "Capture calibration images",
        "cmd": [sys.executable, "-m", "source_code.vision.calibration.run_capture_images"],
    },
    "3": {
        "name": "Run calibration",
        "cmd": [sys.executable, "-m", "source_code.vision.calibration.run_calibration"],
    },
    "4": {
        "name": "Run undistortion",
        "cmd": [sys.executable, "-m", "source_code.vision.undistortion.run_undistort"],
    },
    "5": {
        "name": "Run homography",
        "cmd": [sys.executable, "-m", "source_code.vision.homography.run_homography"],
    },
    "6": {
        "name": "RoboDK Simulation",
        "cmd": [sys.executable, "-m", "source_code.simulation.run_AI_script"],
    },
    "7": {
        "name": "Object segmentation",
        "cmd": [sys.executable, "-m", "source_code.vision.object_segmentation.object_segmentation"],
    },
    "8": {
        "name": "Image to world transformation",
        "cmd": [sys.executable, "-m", "source_code.vision.homography.run_image_to_world_transformation"],
    },
    "9": {
        "name": "Clear existing calibration images",
        "cmd": [sys.executable, "-m", "source_code.vision.camera.clear_raw_images"],
    },
}

# =========================
# MAIN WINDOW
# =========================
root = tk.Tk()
root.title("Physical AI Control Panel")
root.geometry("1100x700")
root.configure(bg="#0f172a")

# =========================
# LOG BOX
# =========================
log_box = tk.Text(root, height=20, bg="#111827", fg="white", font=("Consolas", 11))
log_box.pack(fill="both", expand=True, padx=10, pady=10)

def log(msg):
    log_box.insert("end", msg + "\n")
    log_box.see("end")

# =========================
# RUN COMMAND (NON-BLOCKING)
# =========================
def run_command(cmd_id):
    name, cmd = RUN_COMMANDS[cmd_id]

    def task():
        log(f"\n=== STARTING: {name} ===")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate()

        if stdout:
            log(stdout)

        if stderr:
            log(stderr)

        if process.returncode == 0:
            log(f"FINISHED: {name}")
        else:
            log(f"ERROR in: {name}")

        log("-" * 50)

    threading.Thread(target=task).start()

# =========================
# BUTTON PANEL
# =========================
panel = tk.Frame(root, bg="#0f172a")
panel.pack(fill="x")

def make_btn(text, cmd_id):
    return tk.Button(
        panel,
        text=text,
        command=lambda: run_command(cmd_id),
        bg="#1f2937",
        fg="white",
        font=("Arial", 11),
        padx=10,
        pady=5
    )

# Row 1
for cmd_id, data in RUN_COMMANDS.items():
    tk.Button(
        panel,
        text=data["name"],
        command=lambda c=cmd_id: run_command(c)
    ).pack(side="left", padx=5)

# =========================
# PIPELINE FUNCTIONS (GUI VERSION)
# =========================
def run_full_pipeline():
    steps = ["2", "3", "4", "5"]

    def task():
        log("\n=== FULL PIPELINE START ===")
        for s in steps:
            name, cmd = RUN_COMMANDS[s]
            log(f"Running: {name}")

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = process.communicate()

            if out:
                log(out)
            if err:
                log(err)

            if process.returncode != 0:
                log("PIPELINE STOPPED (ERROR)")
                return

        log("PIPELINE COMPLETE")

    threading.Thread(target=task).start()

def run_simulation_flow():
    def task():
        log("\n=== OBJECT + SIMULATION FLOW ===")

        run_command("7")
        time.sleep(1)
        run_command("8")
        time.sleep(1)
        run_command("6")

    threading.Thread(target=task).start()

# =========================
# EXTRA CONTROL BUTTONS
# =========================
control = tk.Frame(root, bg="#0f172a")
control.pack(fill="x")

tk.Button(
    control,
    text="RUN FULL CALIBRATION PIPELINE",
    bg="#10b981",
    fg="black",
    command=run_full_pipeline
).pack(side="left", padx=10, pady=10)

tk.Button(
    control,
    text="OBJECT → SIMULATION FLOW",
    bg="#3b82f6",
    fg="white",
    command=run_simulation_flow
).pack(side="left", padx=10)

tk.Button(
    control,
    text="CLEAR LOG",
    bg="#ef4444",
    fg="white",
    command=lambda: log_box.delete("1.0", "end")
).pack(side="right", padx=10)

# =========================
# RUN
# =========================
root.mainloop()