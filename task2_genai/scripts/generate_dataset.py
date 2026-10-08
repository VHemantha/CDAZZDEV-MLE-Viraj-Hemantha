"""CLI: generate the synthetic AML triage dataset with the teacher model (resumable).

    python scripts/generate_dataset.py            # from task2_genai/
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'CLI wrapper for resumable teacher generation',
# Date: 2026-10-07

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

from aml_triage import config  # noqa: E402
from aml_triage.generate import generate  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    total = generate()
    print(f"{total} valid cases in {config.RAW_PATH}")
