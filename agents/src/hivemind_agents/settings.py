"""settings: model names from models.toml, everything else from the environment (see example.env)."""

import os
import tomllib
from dataclasses import dataclass
from importlib import resources

BUSINESS_NAME = "HiveMart"
MIN_KEY_BYTES = 32


@dataclass(frozen=True)
class Job:
    name: str
    provider: str
    model: str
    temperature: float


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
        if row["provider"] not in providers:
            raise ValueError(f"job {name!r} names the provider {row['provider']!r}, which models.toml does not list")
        jobs[name] = Job(name=name, provider=row["provider"], model=row["model"],
                         temperature=float(row.get("temperature", 0.0)))
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
