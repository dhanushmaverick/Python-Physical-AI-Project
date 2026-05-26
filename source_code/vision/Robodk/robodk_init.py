from robodk.robolink import *
from robodk.robomath import *

def Robodk_init():
     global RDK
     RDK = Robolink()
     cd = os.getcwd()
     dir = os.path.dirname(__file__)
     station_path = os.path.join(dir,"RoboDKSIM.rdk")
     RDK.AddFile(station_path)
    

def main():
    Robodk_init()
    print("Robodk_station opened")
if __name__ == "__main__":
     main()