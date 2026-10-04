import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src` imports work when running scripts directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import HOTELS_CITIES
from src.scheduler.jobs import start_scheduler


if __name__ == "__main__":
    import asyncio

    # start_scheduler builds an AsyncIOScheduler, which only fires jobs while its
    # event loop runs. The previous `while True: time.sleep(60)` never ran one, so
    # no scheduled job executed (measured 2026-09-15: a 1 s interval job fired 0
    # times in 3.5 s). Create the loop first so the scheduler binds to it.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_scheduler(HOTELS_CITIES)
    print("Scheduler started. Press Ctrl+C to stop.")

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("Scheduler stopped.")
