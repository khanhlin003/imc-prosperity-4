import sys
from argparse import ArgumentParser
from pathlib import Path

import orjson

from prosperity3bt.file_reader import default_data_root


def parse_prices(activities_log: str, output_dir: Path, round_day: str) -> None:
    output_file = output_dir / f"prices_{round_day}.csv"

    print(f"Writing prices data to {output_file}")
    with output_file.open("w+", encoding="utf-8") as f:
        f.write(activities_log + "\n")


def parse_trades(trade_history: str, output_dir: Path, round_day: str) -> None:
    trades = orjson.loads(trade_history)

    output_file = output_dir / f"trades_{round_day}.csv"

    print(f"Writing trades data to {output_file}")
    with output_file.open("w+", encoding="utf-8") as f:
        f.write("timestamp;buyer;seller;symbol;currency;price;quantity\n")

        for t in trades:
            row = ";".join(
                [
                    str(t["timestamp"]),
                    t["buyer"],
                    t["seller"],
                    t["symbol"],
                    t["currency"],
                    str(t["price"]),
                    str(t["quantity"]),
                ]
            )

            f.write(row + "\n")


def main() -> None:
    parser = ArgumentParser(
        description="Extract prices/trades from a submission log into `data/roundN/` (same layout the backtester reads).",
    )
    parser.add_argument("file", type=str, help="path to the log file")
    parser.add_argument("round", type=int, help="round the logs belong to")
    parser.add_argument("day", type=int, help="day the logs belong to")

    args = parser.parse_args()

    file = Path(args.file).expanduser().resolve()
    if not file.is_file():
        print(f"Error: {file} is not a file")
        sys.exit(1)

    logs = file.read_text()

    sections = {}
    for block in logs.split("\n\n"):
        block = block.strip()
        if len(block) == 0:
            continue

        newline_idx = block.index("\n")
        category = block[: newline_idx - 1]
        content = block[newline_idx + 1 :]

        sections[category] = content

    base = default_data_root()
    if base is None:
        base = Path.cwd() / "data"
    base.mkdir(parents=True, exist_ok=True)
    output_dir = base / f"round{args.round}"
    output_dir.mkdir(parents=True, exist_ok=True)

    round_day = f"round_{args.round}_day_{args.day}"

    parse_prices(sections["Activities log"], output_dir, round_day)
    parse_trades(sections["Trade History"], output_dir, round_day)


if __name__ == "__main__":
    main()
