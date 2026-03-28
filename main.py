from pathlib import Path
import sys


# Umożliwia import pakietu z katalogu "src" przy uruchamianiu main.py z katalogu projektu
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from raspi_face_capture.pipeline import CapturePipeline


def main() -> None:
    pipeline = CapturePipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
