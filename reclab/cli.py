import argparse
from pathlib import Path
import json
from .paths import read_json


def main():
    p = argparse.ArgumentParser(description="Local recommendation learning lab")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("archive", type=Path)
    prep.add_argument("--max-items", type=int, default=5000)
    prep.add_argument("--max-users", type=int, default=12000)
    sub.add_parser("baseline")
    fit = sub.add_parser("train")
    fit.add_argument("--config", type=Path)
    fit.add_argument("--run-id")
    test = sub.add_parser("final-test")
    test.add_argument("run_id")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=18580)
    a = p.parse_args()
    if a.command == "prepare":
        from .data import prepare

        print(
            json.dumps(
                prepare(a.archive, a.max_items, a.max_users),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif a.command == "baseline":
        from .data import Dataset
        from .train import ensure_baselines

        print(json.dumps(ensure_baselines(Dataset()), indent=2))
    elif a.command == "train":
        from .train import train, TrainConfig

        report = train(
            TrainConfig(**(read_json(a.config) if a.config else {})), a.run_id
        )
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("samples", "curve")},
                ensure_ascii=False,
                indent=2,
            )
        )
    elif a.command == "final-test":
        from .train import final_test

        print(json.dumps(final_test(a.run_id), indent=2))
    else:
        import uvicorn

        uvicorn.run("reclab.service:app", host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
