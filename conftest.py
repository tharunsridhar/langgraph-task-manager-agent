import os
import tempfile
from pathlib import Path

# Point the app at throwaway databases before any test module imports it.
_TMP = tempfile.mkdtemp()
os.environ.setdefault("TASKS_DB", str(Path(_TMP) / "tasks.db"))
os.environ.setdefault("CHECKPOINT_DB", str(Path(_TMP) / "checkpoints.db"))
# Tests never talk to Opik, whatever the developer has configured.
os.environ["OPIK_TRACING"] = "off"
