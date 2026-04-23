import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src` imports work when running scripts directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import HOTELS_CITIES
from src.scheduler.jobs import start_scheduler


if __name__ == "__main__":
    start_scheduler(HOTELS_CITIES)
    print("Scheduler started. Press Ctrl+C to stop.")

    import time

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Scheduler stopped.")
