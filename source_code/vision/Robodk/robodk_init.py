import os
from robodk.robolink import *

RDK = None

def Robodk_init():
    print("START")
    RDK = Robolink()
    print("CONNECTED")

    dir_path = os.path.dirname(__file__)
    station_path = os.path.join(dir_path, "RoboDKSIM.rdk")

    print("Loading station:", station_path)

    if not os.path.exists(station_path):
        raise FileNotFoundError(station_path)

    RDK.AddFile(station_path)
   

    
        
    return RDK
    




   
