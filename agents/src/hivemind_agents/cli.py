"""hm-agents: run the chat service, make signing keys, check the model names."""

import argparse
import secrets
import sys


def build_service():
    import httpx

    from hivemind_agents import settings, tracing
    from hivemind_agents.capabilities.identity import IdentityVerifier
    from hivemind_agents.chat import ChatService, Deps
    from hivemind_agents.llm import ModelRouter
    from hivemind_agents.store import ChatStore

    endpoint = settings.traces_endpoint()
    tracer = tracing.Tracer(tracing.make_provider(tracing.otlp_exporter(endpoint) if endpoint else None))
    verifier = IdentityVerifier(settings.signing_key("HM_SESSION_SECRET"), settings.signing_key("HM_CASE_TOKEN_SECRET"))
    deps = Deps(
        http=httpx.Client(base_url=settings.backend_url(), timeout=10.0),
        llm=ModelRouter(settings.load_models(), tracer),
        store=ChatStore(settings.agent_conninfo()),
        tracer=tracer,
        verifier=verifier,
    )
    return ChatService(deps)


def app():
    """for `uvicorn --factory hivemind_agents.cli:app`."""
    from hivemind_agents.api import create_app

    return create_app(build_service())


def new_keys() -> str:
    return (f"HM_SESSION_SECRET={secrets.token_urlsafe(48)}\n"
            f"HM_CASE_TOKEN_SECRET={secrets.token_urlsafe(48)}\n")


def check_models() -> int:
    """asks each provider which models the key can use, and says whether every name in models.toml is one."""
    from google import genai

    from hivemind_agents import settings

    models = settings.load_models()
    key = settings.env(models.key_envs["google"])
    client = genai.Client(api_key=key)
    available = {m.name.removeprefix("models/") for m in client.models.list()}
    missing = 0
    for job in models.jobs.values():
        ok = job.model in available
        missing += not ok
        print(f"{'ok     ' if ok else 'MISSING'} {job.name}: {job.model}")
    if missing:
        print(f"{missing} model name(s) are not offered to this key. the key can use: " + ", ".join(sorted(available)))
    return 1 if missing else 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="hm-agents")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the chat service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    sub.add_parser("new-keys", help="print two new signing keys to paste into .env")
    sub.add_parser("check-models", help="check the model names in models.toml against the provider (calls google)")
    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        uvicorn.run(app(), host=args.host, port=args.port)
    elif args.command == "new-keys":
        sys.stdout.write(new_keys())
    elif args.command == "check-models":
        sys.exit(check_models())


if __name__ == "__main__":
    main()
