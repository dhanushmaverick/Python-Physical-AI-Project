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
def read_roboDK_function_list()->str:
    """String containing python-RoboDK functions which are to be used in order to generate code. Do not use any other roboDK functions except these."""
    with open("source_code/AI/robodk_python_functions.txt", "r") as f:
        commands = f.read()
        return commands