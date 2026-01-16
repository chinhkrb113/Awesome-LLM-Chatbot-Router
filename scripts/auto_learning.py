import argparse
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.learning_auto import maybe_run_learning_loop, load_learning_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-run the learning loop based on smart rules.")
    parser.add_argument("--config", default="config/learning_loop.yaml")
    parser.add_argument("--loop", action="store_true", help="Run continuously on an interval.")
    args = parser.parse_args()

    if not args.loop:
        result = maybe_run_learning_loop(args.config)
        print(result)
        return

    while True:
        result = maybe_run_learning_loop(args.config)
        print(result)
        cfg = load_learning_config(args.config)
        sleep_seconds = int(cfg.get("check_interval_minutes", 60)) * 60
        time.sleep(max(60, sleep_seconds))


if __name__ == "__main__":
    main()
