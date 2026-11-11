"""hm-db: database setup and the openapi contract file."""

import argparse
import json
from pathlib import Path

CONTRACT = Path(__file__).resolve().parents[2] / "openapi.json"


def contract_text() -> str:
    from hivemind_backend.app import app

    return json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(prog="hm-db")
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("migrate", help="create the roles and apply the sql files")
    m.add_argument("--reset", action="store_true", help="drop the database first (all data is lost)")
    m.add_argument("--db", help="database name (default: POSTGRES_DB)")
    sub.add_parser("export-openapi", help="write backend/openapi.json from the code")
    args = parser.parse_args()

    if args.command == "migrate":
        from hivemind_backend.migrate import migrate

        applied = migrate(args.db, reset=args.reset)
        print("applied: " + (", ".join(applied) if applied else "nothing new"))
    elif args.command == "export-openapi":
        CONTRACT.write_text(contract_text(), encoding="utf-8", newline="\n")
        print(f"wrote {CONTRACT}")


if __name__ == "__main__":
    main()
