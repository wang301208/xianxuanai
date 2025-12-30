import os
from pathlib import Path

try:  # Prefer installed autogpt-forge package when available
    from forge.agent import ForgeAgent  # type: ignore
    from forge.sdk import LocalWorkspace  # type: ignore
except ModuleNotFoundError:  # Fallback to monorepo package layout
    from .agent import ForgeAgent
    from .sdk import LocalWorkspace

from .db import ForgeDatabase

default_db_path = Path(os.getenv("DATABASE_PATH", "data/forge.db")).as_posix()
database_name = os.getenv("DATABASE_STRING") or f"sqlite:///{default_db_path}"
workspace_root = os.getenv("AGENT_WORKSPACE") or str(Path("data") / "workspace")
workspace = LocalWorkspace(workspace_root)
database = ForgeDatabase(database_name, debug_enabled=False)
agent = ForgeAgent(database=database, workspace=workspace)

app = agent.get_agent_app()

try:  # Optional autonomous WholeBrain loop (enabled by default; disable via AUTONOMY_ENABLED=false)
    from .autonomy import attach_autonomy
except Exception:  # pragma: no cover - best effort runtime wiring
    attach_autonomy = None  # type: ignore[assignment]

if attach_autonomy is not None:
    attach_autonomy(app)
