import os
import json

from core.config import PROGRESS_FILE


def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {"total_target": 0, "generated_so_far": 0}
    try:
        with open(PROGRESS_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                return {"total_target": 0, "generated_so_far": 0}
            return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        # File may be partially written due to concurrent access
        return {"total_target": 0, "generated_so_far": 0}


def save_progress(total_target, generated_so_far):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(
            {
                "total_target": total_target,
                "generated_so_far": generated_so_far
            },
            f,
            indent=2
        )
