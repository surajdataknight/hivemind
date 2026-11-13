"""generate(seed, config) -> every table's rows, business and truth."""

from hivemind_datagen.faults import FaultInjector
from hivemind_datagen.generate import Config, Generator, dataset_sha256


def generate(seed: int, cfg: Config = Config()) -> dict[str, list[dict]]:
    gen = Generator(seed, cfg)
    gen.build_catalog()
    gen.build_stores()
    gen.build_riders()
    gen.build_customers()
    gen.build_offers()
    gen.build_natural_stockouts()
    gen.build_restocks()
    gen.build_orders()
    gen.build_fraud_rings()
    FaultInjector(gen).run()
    gen.build_past_claims({f["order_id"] for f in gen.t["truth.order_faults"]})
    gen.build_stock_levels()
    gen.add("truth.generation_runs", {"run_id": f"RUN-{seed}", "seed": seed, "config": cfg.to_json(),
                                      "data_sha256": dataset_sha256(gen.t), "generated_at": cfg.as_of})
    return gen.t
