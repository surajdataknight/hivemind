"""settings: model names from models.toml, everything else from the environment (see example.env)."""

import os
import tomllib
from dataclasses import dataclass
from importlib import resources

BUSINESS_NAME = "HiveMart"
MIN_KEY_BYTES = 32
TIMEOUT_S = 20.0  # a job's model call limit when models.toml gives none


@dataclass(frozen=True)
class Job:
    name: str
    provider: str
    model: str
    temperature: float
    timeout_s: float = TIMEOUT_S
    retry_timeouts: bool = True  # false: one timeout ends the call (a slow model is not asked again)
    total_s: float | None = None  # all tries of one call together, waits included; none: no shared limit
    fallback: str | None = None   # the model asked at once when this job's model says it is busy (record 013)
    fallback_provider: str | None = None  # the fallback's provider; none: the job's own
    busy_waits: tuple[float, ...] | None = None  # seconds before each retry after a busy answer; none: the router's


@dataclass(frozen=True)
class ModelSettings:
    jobs: dict[str, Job]
    key_envs: dict[str, str]  # provider -> environment variable that holds its key

    def job(self, name: str) -> Job:
        if name not in self.jobs:
            raise KeyError(f"no model is set for the job {name!r} in models.toml")
        return self.jobs[name]


def load_models(text: str | None = None) -> ModelSettings:
    if text is None:
        text = (resources.files("hivemind_agents") / "models.toml").read_text(encoding="utf-8")
    data = tomllib.loads(text)
    providers = data.get("providers", {})
    jobs = {}
    for name, row in data.get("jobs", {}).items():
        for key in ("provider", "fallback_provider"):
            if key in row and row[key] not in providers:
                raise ValueError(f"job {name!r} names the provider {row[key]!r}, which models.toml does not list")
        jobs[name] = Job(name=name, provider=row["provider"], model=row["model"],
                         temperature=float(row.get("temperature", 0.0)),
                         timeout_s=float(row.get("timeout_s", TIMEOUT_S)),
                         retry_timeouts=bool(row.get("retry_timeouts", True)),
                         total_s=float(row["total_s"]) if "total_s" in row else None,
                         fallback=row.get("fallback"), fallback_provider=row.get("fallback_provider"),
                         busy_waits=tuple(float(w) for w in row["busy_waits"]) if "busy_waits" in row else None)
    return ModelSettings(jobs=jobs, key_envs={name: row["key_env"] for name, row in providers.items()})


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name) or default
    if not value:
        raise RuntimeError(f"{name} is not set; see example.env and run with `uv run --env-file .env ...`")
    return value


def signing_key(name: str) -> str:
    key = env(name)
    if len(key.encode()) < MIN_KEY_BYTES:
        raise RuntimeError(f"{name} is shorter than {MIN_KEY_BYTES} characters; make a new pair with "
                           "`hm-agents new-keys`")
    return key


def panel_key() -> str | None:
    """the scenario panel's key, or None when it is not set (the panel is then off)."""
    key = os.environ.get("HM_PANEL_KEY")
    if key and len(key.encode()) < MIN_KEY_BYTES:
        raise RuntimeError(f"HM_PANEL_KEY is shorter than {MIN_KEY_BYTES} characters; make one with "
                           "`hm-agents new-keys`")
    return key or None


def backend_url() -> str:
    return env("HM_BACKEND_URL", "http://127.0.0.1:8080")


def traces_endpoint() -> str | None:
    """where traces go (phoenix in docker compose). `off` switches tracing off."""
    value = os.environ.get("HM_TRACES_ENDPOINT") or "http://127.0.0.1:6006/v1/traces"
    return None if value.lower() == "off" else value


def agent_conninfo() -> str:
    from psycopg.conninfo import make_conninfo

    pw = env("HM_AGENT_PASSWORD")
    return make_conninfo(
        host=os.environ.get("POSTGRES_HOST", "127.0.0.1"),
        port=os.environ.get("POSTGRES_PORT", "5433"),
        user="hm_agent",
        password=pw,
        dbname=os.environ.get("HM_DB_NAME") or env("POSTGRES_DB"),
        options="-c TimeZone=UTC",
    )
