from __future__ import annotations

import argparse
import logging

from .config import load_config
from .db import DB


def main() -> None:
    p = argparse.ArgumentParser(prog="router")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    st = sub.add_parser("stats")
    st.add_argument("--days", type=int, default=7)
    args = p.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "serve":
        import uvicorn

        from .app import create_app

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
        uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port)
    else:
        rows = DB(cfg.db_path).stats(args.days)
        print(f"{'day':<11}{'route':<18}{'n':>5}{'prompt':>10}{'compl':>9}{'cost_usd':>10}{'avg_ms':>9}")
        for day, route, n, pt, ct, cost, lat in rows:
            print(f"{day:<11}{route or '-':<18}{n:>5}{pt:>10}{ct:>9}{cost:>10.4f}{lat:>9.0f}")


if __name__ == "__main__":
    main()
