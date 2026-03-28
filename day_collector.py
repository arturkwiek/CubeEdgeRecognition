"""Day Collector: cienki wrapper uruchamiający dzienny proces zbierania twarzy.

Alias do głównego pipeline'u z pliku main.py.
"""

from __future__ import annotations

from main import main as run_day_collector


if __name__ == "__main__":
    run_day_collector()
