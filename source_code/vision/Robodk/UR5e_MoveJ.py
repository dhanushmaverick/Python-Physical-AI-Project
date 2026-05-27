from robodk_init import RDK
import time

def UR5e_MoveJ(Target):
    global RDK
    robot = RDK.Item("UR5e")
    Target = RDK.Item(Target)
    robot.setPoseFrame(Target.Parent())
    robot.MoveJ(Target)
    time.sleep(0.5)