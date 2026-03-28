"""Night Trainer: cienki wrapper uruchamiający nocny proces trenowania.

Alias do pipeline'u z nightly.py.
"""

from __future__ import annotations

from nightly import run_night_trainer


if __name__ == "__main__":
    run_night_trainer()
