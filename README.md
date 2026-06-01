# Physical AI Enabled Vision Pick-and-Place Simulator

A vision-guided robotic manipulation framework that combines **Computer Vision**, **Workspace Calibration**, **AI-Based Task Planning**, and **RoboDK Simulation** to enable natural-language control of a UR5e pick-and-place system.

---

## Overview

Traditional robotic pick-and-place systems require engineers to manually write motion programs whenever object arrangements change. This project explores a simpler workflow where users describe a desired arrangement in natural language and allow an AI agent to generate the required robot motion plan automatically.

The system integrates:

- Camera Calibration
- Workspace Calibration (Homography)
- Object Segmentation
- Image-to-World Coordinate Transformation
- AI Motion Planning
- RoboDK Simulation

The result is a proof-of-concept **Physical AI pipeline** that removes the need for repeated robot programming for simple rearrangement tasks.

---

## Key Features

### Vision-Based Object Detection

Detects:

- Red Block
- Green Block
- Blue Block

using color segmentation and contour-based blob detection.

### Workspace Calibration

Converts image coordinates into robot world coordinates through homography estimation.

### Natural Language Robot Programming

Example commands:

```text
Stack the red block on the blue block.
```

```text
Place the green block on the red block and then place the blue block on top.
```

The AI planner converts these instructions into robot motion code.

### RoboDK Integration

Automatically:

- Opens RoboDK
- Loads the simulation workspace
- Places blocks in detected locations
- Executes generated robot motion plans

### GUI-Based Workflow

Users interact through a graphical interface instead of the terminal.

---

# End User Guide

## What the System Does

The system automatically:

1. Detects objects in the workspace
2. Determines their real-world positions
3. Interprets your natural-language instruction
4. Generates a robot motion plan
5. Executes the task in RoboDK

---

## Supported Objects

The current version supports:

- Red Block
- Green Block
- Blue Block

---

## Requirements

Before running the project, install:

- Python 3.13+
- RoboDK
- OpenAI API Key

---

## Installation

### Activate Virtual Environment

```powershell
.\environment\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install -r dependencies.txt
```

### Set OpenAI API Key

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

---

## Running the Application

Launch the graphical interface:

```powershell
python -m source_code.gui
```

---

## First-Time Setup

### Camera Calibration

Run this when:

- Using a new camera
- Camera parameters have changed

### Workspace Calibration

Run this when:

- The workspace has moved
- The camera position relative to the workspace has changed

---

## Typical Workflow

### Step 1: Open the Application

Launch the GUI.

### Step 2: Calibrate (If Required)

- Calibrate Camera
- Calibrate Workspace

### Step 3: Enter a Task

Example:

```text
Create a stack with blue at the bottom, red in the middle, and green on top.
```

### Step 4: Run Simulation

Click **Simulate**.

The system will:

```text
Detect Objects
      ↓
Compute World Coordinates
      ↓
Generate AI Motion Plan
      ↓
Execute RoboDK Simulation
```

---

## GUI Functions

### Calibrate Camera

Performs camera calibration and image correction.

### Calibrate Workspace

Generates workspace mapping between image coordinates and robot coordinates.

### User Query

Enter a natural-language task description.

### Simulate

Runs the complete pick-and-place pipeline.

---

# Technical Documentation

## System Architecture

```text
┌─────────────────────────────┐
│          USER GUI           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Camera Calibration Module   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Workspace Calibration       │
│ (Homography Generation)     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Object Segmentation         │
│ (Red / Green / Blue Blocks) │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Image → World Conversion    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ AI Motion Planner           │
│ Natural Language → Code     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ RoboDK Simulation           │
│ UR5e Pick & Place           │
└─────────────────────────────┘
```

---

## Workspace Assumptions

### Robot

- UR5e Manipulator

### Camera

- Fixed overhead camera mounted above the workspace

### Workspace

- Flat tabletop environment

### Calibration Tool

- Checkerboard calibration rig

---

## Project Structure

```text
source_code/
│
├── gui.py
├── main.py
├── RoboDKSIM.rdk
│
├── AI/
│   ├── main_motion_plan.py
│   ├── open_ai_api.py
│   └── tools.py
│
├── simulation/
│   ├── RoboDK_config.py
│   ├── run_AI_script.py
│   └── example_main_motion_plan.py
│
├── utility/
│   ├── world_coords.py
│   └── paths.py
│
└── vision/
    ├── camera/
    ├── calibration/
    ├── homography/
    ├── object_segmentation/
    └── undistortion/
```

---

## Calibration Pipeline

### Camera Calibration

The purpose of camera calibration is to estimate camera intrinsic parameters that are later used for image undistortion and accurate coordinate estimation.

#### Process

```text
Capture Checkerboard Images
            ↓
Camera Calibration
            ↓
Save Intrinsic Parameters
```

#### Outputs

```text
camera_intrinsics.npz
calibration_report.json
```

---

### Workspace Calibration

Workspace calibration computes the homography transformation between image coordinates and world coordinates.

#### Calibration Requirements

A planar checkerboard calibration rig is required.

#### Calibration Rules

> **Rule 1:** All four calibration corners must remain visible.

> **Rule 2:** The calibration board may be translated or rotated.

> **Rule 3:** One calibration corner should coincide with the desired world origin.

```text
World Origin = (0,0)
```

> **Rule 4:** When selecting calibration points, begin at the origin and proceed anti-clockwise.

```text
Start at Origin
      ↓
Move Anti-Clockwise
```

> **Rule 5:** The usable workspace is restricted to the camera view from the robot home position.

#### Outputs

```text
World_Pose.json
homography_report.json
```

---

## Image Undistortion

Uses intrinsic calibration parameters to remove lens distortion before object detection and coordinate estimation.

```text
Raw Image
    ↓
Undistortion
    ↓
Corrected Image
```

---

## Object Detection Pipeline

The object segmentation pipeline identifies colored blocks in the workspace.

### Supported Objects

- Red Block
- Green Block
- Blue Block

### Processing Pipeline

```text
Image Capture
      ↓
Color Thresholding
      ↓
Binary Masks
      ↓
Blob Detection
      ↓
Centroid Extraction
```

### Detection Parameters

#### Color Thresholds

```python
r_thresh_val
g_thresh_val
b_thresh_val
```

Used to generate binary masks.

#### Blob Filtering

```python
area_min
area_max
```

Used to reject noise and invalid detections.

---

## Image-to-World Transformation

Detected object locations are converted into world coordinates using the homography matrix.

```text
Image Coordinates
        ↓
Homography Matrix
        ↓
World Coordinates
```

### Outputs

```text
Pose.json
World_Pose.json
```

---

## AI Motion Planning

The AI planning module converts user instructions into executable robot motion code.

### Inputs

#### Detected Block Coordinates

Example:

```json
{
  "red": [x, y],
  "green": [x, y],
  "blue": [x, y]
}
```

#### User Prompt

Example:

```text
Place green on blue and red on top.
```

#### Reference Motion Plans

Example motion plans are provided to guide generation.

### Output

The AI generates motion plans using a predefined set of approved robot functions.

This design ensures:

- Consistent robot behavior
- Safer execution
- Predictable simulation results

---

## RoboDK Simulation

The simulation module automatically:

1. Launches RoboDK
2. Loads the UR5e workstation
3. Places blocks at detected coordinates
4. Executes AI-generated robot motion code

Main simulation files:

```text
simulation/
├── RoboDK_config.py
├── run_AI_script.py
└── example_main_motion_plan.py
```

---

## Simulation Pipeline

```text
Object Segmentation
        ↓
Image-to-World Conversion
        ↓
AI Motion Planning
        ↓
RoboDK Simulation
```

---

## Research Contribution

This project demonstrates a complete Physical AI workflow where:

```text
Computer Vision
       +
Calibration
       +
Spatial Reasoning
       +
Large Language Models
       +
Robot Simulation
```

are integrated into a single user-friendly application.

The primary contribution is the elimination of repetitive robot programming by enabling natural-language task specification while preserving geometric accuracy through camera calibration and homography-based coordinate transformation.

---

# Authors

- Dhanush Vulli Bala
- Praneeth Manickam Srinivas
- Joel George Thomas

---

**Physical AI Enabled Vision Pick-and-Place Simulator**

A proof-of-concept system demonstrating the integration of Computer Vision, AI Planning, and Robotic Manipulation through a simplified user experience.