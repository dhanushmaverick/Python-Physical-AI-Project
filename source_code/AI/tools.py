# We will be looking at how to use 3 tools
from langchain.tools import tool
from source_code.utility.world_coords import *

@tool 
def save_to_py(data:str)->str:
    """Save the generated python-robodk code to a .py file named source_code/AI/main_motion_plan.py"""
    with open("source_code/AI/main_motion_plan.py", "w") as f:
        f.write(data)
    
    return f"Saved code to source_code/AI/main_motion_plan.py"

@tool 
def get_world_coords_string()->str:
    """String containing current world coordinates of blocks is given through this tool. Everytime block coordinates are needed, run this."""
    return structured_string

@tool
def read_roboDKContext()->str:
    """String containing RoboDKContext class definition code (from source_code/simulation/RoboDK_config.py) which is to be used in order to generate code. Do not use any other roboDK functions or create any own functions except the member functions of this class."""
    with open("source_code/simulation/RoboDK_config.py", "r") as f:
        return f.read()

@tool
def read_exemplary_code()->str:
    """String containing example code (from source_code/simulation/example_main_motion_plan.py) of what is to be generated. Do not create or use any other functions except the ones used here"""
    with open("source_code/simulation/example_main_motion_plan.py", "r") as f:
        return f.read()