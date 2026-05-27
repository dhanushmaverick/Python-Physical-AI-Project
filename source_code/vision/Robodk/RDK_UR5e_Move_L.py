from robodk_init import RDK
import time

def RDK_UR5e_MoveL(Target):
    global RDK
    robot = RDK.Item("UR5e")
    Target = RDK.Item(Target)
    robot.setPoseFrame(Target.Parent())
    robot.MoveL(Target)
    time.sleep(0.5)

