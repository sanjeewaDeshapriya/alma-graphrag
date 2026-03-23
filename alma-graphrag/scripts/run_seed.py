import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from kg_builder.seed_data import seed_all

if __name__ == "__main__":
    seed_all()
    print("Seed data loaded.")
