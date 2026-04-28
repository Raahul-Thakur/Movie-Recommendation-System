from pathlib import Path

import gdown


DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/16baTs2GgiBKA0ho1WiGtpr9M4Woz8BFw?usp=sharing"
REQUIRED_MODEL = Path("hybrid_recommender.pkl")


def main():
    if REQUIRED_MODEL.exists():
        print(f"{REQUIRED_MODEL} already exists, skipping download.")
        return

    gdown.download_folder(
        url=DRIVE_FOLDER_URL,
        output=".",
        quiet=False,
        use_cookies=False,
    )

    if not REQUIRED_MODEL.exists():
        raise FileNotFoundError(
            f"Expected {REQUIRED_MODEL} after downloading from Google Drive."
        )


if __name__ == "__main__":
    main()
