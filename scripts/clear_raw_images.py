from app.utils.paths import RAW_IMAGES_DIR, ensure_directories


def clear_raw_images() -> None:
    ensure_directories()

    files = list(RAW_IMAGES_DIR.glob("*"))

    if not files:
        print("Raw images directory is already empty.")
        return

    for file in files:
        if file.is_file():
            file.unlink()

    print(f"Deleted {len(files)} files from {RAW_IMAGES_DIR}")


if __name__ == "__main__":
    clear_raw_images()