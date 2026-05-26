#run command:
#       python -m source_code.utility.world_coords

#STRUCTURED_STRING CONTAINS WORLD_POSE AS STRING
#USE THE FUNCTION get_world_coords_string() TO GET IT AS STRING WHEN IT WOULD BE IN NEEDED IN AI PART


from source_code.utility.paths import *
import json


with open(HOMOGRAPHY_DATA / "World_Pose.json", "r") as file:
    world_data = json.load(file)

structured_string = json.dumps(world_data)


#print("\n\n",structured_string)   #just for checking how it got printed ... change it however you need

#print(len(structured_string))

