"""hm-seed: generate the made-up data from a seed and load it into postgres."""

import argparse
from collections import Counter

from hivemind_datagen.dataset import generate
from hivemind_datagen.generate import SMALL, Config


def main() -> None:
    parser = argparse.ArgumentParser(prog="hm-seed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small", action="store_true", help="3 days, 60 customers, 130 orders a day, 3 faults per type")
    parser.add_argument("--db", help="database name (default: POSTGRES_DB)")
    parser.add_argument("--dry-run", action="store_true", help="generate and print counts, load nothing")
    args = parser.parse_args()

    cfg = SMALL if args.small else Config()
    tables = generate(args.seed, cfg)
    run = tables["truth.generation_runs"][0]
    faults = Counter(f"{f['fault_type']}/{f['variant']}" for f in tables["truth.order_faults"])
    print(f"seed {args.seed}: {len(tables['orders.orders'])} orders, {len(tables['customer.customers'])} customers, "
          f"{len(tables['truth.order_faults'])} faults")
    for name, count in sorted(faults.items()):
        print(f"  {name}: {count}")
    print(f"data sha256 {run['data_sha256']}")
    if not args.dry_run:
        from hivemind_datagen.load import load

        load(tables, args.db)
        print("loaded")


if __name__ == "__main__":
    main()
