#run command:
#       python -m source_code.simulation.run_AI_script


import importlib.util
from .RoboDK_config import RoboDKContext
from source_code.utility.paths import *


def load_ai_motion_script():
    
    ai_script_path = ROBODK_DIR / "test_script.py"

    if not ai_script_path.exists():
        raise FileNotFoundError(f"AI motion script not found:\n{ai_script_path}")

    spec = importlib.util.spec_from_file_location(
        "main_motion_plan",
        ai_script_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load AI motion script:\n{ai_script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "run_motion"):
        raise RuntimeError(
            "AI motion script must define:\n\n"
            "def run_motion(ctx):\n"
            "    ..."
        )

    return module


def main():
    print("[INFO] Starting RoboDK AI runner...")

    # 1. Open RoboDK station and prepare context
    ctx = RoboDKContext()

    # 2. Update RoboDK block positions from JSON
    print("[INFO] Updating RoboDK block poses from JSON:")
    print(WORLD_POSE)
    ctx.place_blocks_from_json(WORLD_POSE)

    # 3. Load AI-generated motion script
    ai_motion_script = load_ai_motion_script()

    # 4. Execute AI motion script
    print("[INFO] Running AI-generated motion script...")
    ai_motion_script.run_motion(ctx)

    print("[SUCCESS] RoboDK AI runner finished.")


if __name__ == "__main__":
    main()