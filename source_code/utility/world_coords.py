#run command:
#       python -m source_code.utility.world_coords

#STRUCTURED_STRING CONTAINS WORLD_POSE AS STRING
#USE THE FUNCTION get_world_coords_string() TO GET IT AS STRING WHEN IT WOULD BE IN NEEDED IN AI PART


from source_code.utility.paths import *
import json


with open(HOMOGRAPHY_DATA / "World_Pose.json", "r") as file:
    world_data = json.load(file)

structured_string = 'Syntax: "color_block_position" : [x, y], "color_block_orientation" : theta, \n' +json.dumps(world_data)

print("-"*20,"\n\n")
print(structured_string)   #just for checking how it got printed ... change it however you need
print("-"*20,"\n\n")

def get_world_coords_string():
    return structured_string