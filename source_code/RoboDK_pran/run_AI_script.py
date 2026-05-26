import importlib.util
from source_code.utility.paths import *

from .RoboDK_runner import RoboDKRunner


def load_ai_script():
    
    ai_script_path = ROBODK_DIR / "test_script.py"

    if not ai_script_path.exists():
        raise FileNotFoundError(f"AI script not found: {ai_script_path}")

    spec = importlib.util.spec_from_file_location("main_motion_plan", ai_script_path)
    module = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError("Could not load AI script.")

    spec.loader.exec_module(module)

    if not hasattr(module, "run_task"):
        raise RuntimeError("AI script must contain a function: run_task(robot)")

    return module


def main():
    robot = RoboDKRunner()
    ai_script = load_ai_script()

    ai_script.run_task(robot)


if __name__ == "__main__":
    main()