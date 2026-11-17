"""Train the paper-style DAN on the six downloaded NSRDB GHI datasets.

PyCharm usage: open this file and click Run. With no command-line arguments it
automatically trains Dataset1--Dataset6 from data/nsrdb_2022.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import date
from pathlib import Path

import numpy as np

from dan import DAN
from data import (
    download_nsrdb_csv,
    fit_scaler,
    make_supervised,
    read_nsrdb_ghi,
    select_paper_window,
    synthetic_irradiance,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "nsrdb_2022"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "nsrdb_2022"

PAPER_CONFIGS = {
    1: {"delay": 5, "embedding_dim": 5, "kappa": 5, "branches": 15, "theta": 0.5, "lr": 1e-4, "units": 2},
    2: {"delay": 4, "embedding_dim": 5, "kappa": 5, "branches": 5, "theta": 0.7, "lr": 2e-4, "units": 3},
    3: {"delay": 5, "embedding_dim": 5, "kappa": 5, "branches": 5, "theta": 0.5, "lr": 1e-4, "units": 3},
    4: {"delay": 4, "embedding_dim": 4, "kappa": 7, "branches": 7, "theta": 0.6, "lr": 1.5e-4, "units": 2},
    5: {"delay": 5, "embedding_dim": 5, "kappa": 5, "branches": 8, "theta": 0.7, "lr": 1e-4, "units": 2},
    6: {"delay": 5, "embedding_dim": 5, "kappa": 8, "branches": 11, "theta": 0.5, "lr": 1e-4, "units": 4},
}

DOWNLOADED_DATASETS = {
    1: {"city": "Havana", "file": "dataset1_havana_2022.csv"},
    2: {"city": "Mexico City", "file": "dataset2_mexico_city_2022.csv"},
    3: {"city": "Santa Cruz", "file": "dataset3_santa_cruz_2022.csv"},
    4: {"city": "Paramaribo", "file": "dataset4_paramaribo_2022.csv"},
    5: {"city": "Caracas", "file": "dataset5_caracas_2022.csv"},
    6: {"city": "Phoenix", "file": "dataset6_phoenix_2022.csv"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DAN on downloaded NSRDB GHI")
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--all", action="store_true", help="train all six downloaded datasets")
    source.add_argument("--synthetic", action="store_true", help="use offline synthetic irradiance")
    source.add_argument("--csv", type=Path, help="train one dataset from an NSRDB CSV")
    source.add_argument("--download", action="store_true", help="download one location/year and train it")

    parser.add_argument("--dataset", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2022, 1, 1))
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument("--year", type=int)
    parser.add_argument("--api-key", default=os.getenv("NSRDB_API_KEY"))
    parser.add_argument("--email", default=os.getenv("NSRDB_EMAIL"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--soma-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    rmse = float(np.sqrt(np.mean(residual**2)))
    mean_target = float(np.mean(target))
    correlation = float(np.corrcoef(target, prediction)[0, 1])
    return {
        "rmse": rmse,
        "nrmse": rmse / mean_target if mean_target else float("nan"),
        "mae": float(np.mean(np.abs(residual))),
        "r": correlation,
    }


def save_predictions(path: Path, target: np.ndarray, prediction: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "target_ghi_w_m2", "predicted_ghi_w_m2", "error_w_m2"])
        writer.writerows(
            (index, actual, forecast, forecast - actual)
            for index, (actual, forecast) in enumerate(zip(target, prediction))
        )


def save_loss_history(path: Path, losses: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "training_loss_scaled"])
        writer.writerows((epoch, loss) for epoch, loss in enumerate(losses, start=1))


def train_one(
    *,
    dataset_number: int,
    series: np.ndarray,
    source_description: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    soma_threshold: float,
    city: str | None = None,
) -> dict:
    config = PAPER_CONFIGS[dataset_number]
    output_dir.mkdir(parents=True, exist_ok=True)

    x, target = make_supervised(
        series,
        delay=config["delay"],
        embedding_dim=config["embedding_dim"],
    )
    train_size = 1400
    if len(x) <= train_size:
        raise ValueError(
            f"Dataset{dataset_number}: reconstructed samples must exceed "
            f"{train_size}, got {len(x)}"
        )
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = target[:train_size], target[train_size:]

    # The paper does not define its normalization implementation. Fitting the
    # scaler on training data only avoids leakage from the chronological test set.
    scaler = fit_scaler(x_train, y_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    y_train_scaled = scaler.transform(y_train)

    model = DAN(
        input_dim=config["embedding_dim"],
        branches=config["branches"],
        dendritic_units=config["units"],
        kappa=config["kappa"],
        theta_init=config["theta"],
        soma_threshold=soma_threshold,
        seed=seed,
    )

    started = time.perf_counter()
    history = model.fit(
        x_train_scaled,
        y_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=config["lr"],
        seed=seed,
        verbose_every=max(1, epochs // 10),
    )
    elapsed_seconds = time.perf_counter() - started

    prediction = scaler.inverse_transform(model.predict(x_test_scaled))
    scores = metrics(y_test, prediction)

    model.save(output_dir / "dan_model.npz")
    save_predictions(output_dir / "predictions.csv", y_test, prediction)
    save_loss_history(output_dir / "loss_history.csv", history.losses)

    report = {
        "dataset": dataset_number,
        "city": city,
        "source": source_description,
        "series_points": int(len(series)),
        "reconstructed_samples": int(len(x)),
        "training_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "config": config,
        "assumptions": {
            "forecast_horizon": "one step ahead",
            "soma_threshold_beta": soma_threshold,
            "normalization": "train-only min-max",
            "scaler_min_ghi": scaler.minimum,
            "scaler_max_ghi": scaler.maximum,
            "optimizer": "bias-corrected AMSGrad, beta1=0.9, beta2=0.999, epsilon=1e-8",
            "initialization": "w~N(2*theta,0.25), delta~U(0.8,1.2), normalized lambda, varsigma=0.5",
            "batch_size": batch_size,
            "seed": seed,
        },
        "metrics": scores,
        "final_training_loss": history.losses[-1],
        "elapsed_seconds": elapsed_seconds,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def train_all_downloaded(args: argparse.Namespace) -> None:
    output_root = args.output_dir or DEFAULT_OUTPUT_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    failures: list[str] = []

    print("DAN 六城市 NSRDB 实验")
    print(f"输入目录: {args.data_dir.resolve()}")
    print(f"输出目录: {output_root.resolve()}")
    print(f"epochs={args.epochs}, batch_size={args.batch_size}, seed={args.seed}")

    for dataset_number, dataset_info in DOWNLOADED_DATASETS.items():
        csv_path = args.data_dir / dataset_info["file"]
        city_slug = csv_path.stem.replace(f"dataset{dataset_number}_", "").replace("_2022", "")
        dataset_output = output_root / f"dataset{dataset_number}_{city_slug}"
        print()
        print("=" * 72)
        print(f"Dataset{dataset_number}: {dataset_info['city']}")
        print(f"CSV: {csv_path}")

        try:
            if not csv_path.exists():
                raise FileNotFoundError(f"missing downloaded CSV: {csv_path}")
            records = read_nsrdb_ghi(csv_path)
            series = select_paper_window(records, start_date=args.start_date)
            print(f"读取完成：全年 {len(records)} 条，论文窗口 {len(series)} 点。")
            print(f"论文参数：{PAPER_CONFIGS[dataset_number]}")
            report = train_one(
                dataset_number=dataset_number,
                series=series,
                source_description=str(csv_path.resolve()),
                output_dir=dataset_output,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                soma_threshold=args.soma_threshold,
                city=dataset_info["city"],
            )
            reports.append(report)
            score = report["metrics"]
            print(
                "完成："
                f"RMSE={score['rmse']:.3f}, nRMSE={score['nrmse']:.4f}, "
                f"MAE={score['mae']:.3f}, R={score['r']:.4f}, "
                f"耗时={report['elapsed_seconds']:.1f}s"
            )
        except Exception as exc:
            message = f"Dataset{dataset_number} {dataset_info['city']}: {exc}"
            failures.append(message)
            print(f"失败：{message}")

    summary_json = output_root / "summary.json"
    summary_json.write_text(
        json.dumps({"results": reports, "failures": failures}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_csv = output_root / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["dataset", "city", "train_samples", "test_samples", "rmse", "nrmse", "mae", "r", "seconds"]
        )
        for report in reports:
            score = report["metrics"]
            writer.writerow(
                [
                    report["dataset"],
                    report["city"],
                    report["training_samples"],
                    report["test_samples"],
                    score["rmse"],
                    score["nrmse"],
                    score["mae"],
                    score["r"],
                    report["elapsed_seconds"],
                ]
            )

    print()
    print("=" * 72)
    print(f"成功：{len(reports)}/6；失败：{len(failures)}/6")
    print(f"汇总结果：{summary_csv}")
    if failures:
        raise SystemExit(1)


def train_single(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or (PROJECT_DIR / "outputs" / "single_run")
    if args.synthetic:
        series = synthetic_irradiance(seed=args.seed)
        source_description = "synthetic smoke-test data"
    else:
        csv_path = args.csv
        if args.download:
            missing = [
                name
                for name in ("latitude", "longitude", "year", "api_key", "email")
                if getattr(args, name) is None
            ]
            if missing:
                raise SystemExit(
                    "--download also requires: "
                    + ", ".join("--" + name.replace("_", "-") for name in missing)
                )
            csv_path = output_dir / f"nsrdb_{args.latitude}_{args.longitude}_{args.year}.csv"
            download_nsrdb_csv(
                csv_path,
                api_key=args.api_key,
                email=args.email,
                latitude=args.latitude,
                longitude=args.longitude,
                year=args.year,
            )
        if csv_path is None:
            raise ValueError("single-dataset mode requires --csv, --download, or --synthetic")
        records = read_nsrdb_ghi(csv_path)
        series = select_paper_window(records, start_date=args.start_date)
        source_description = str(csv_path.resolve())

    report = train_one(
        dataset_number=args.dataset,
        series=series,
        source_description=source_description,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        soma_threshold=args.soma_threshold,
    )
    print(json.dumps(report["metrics"], indent=2))
    print(f"Artifacts written to {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    # PyCharm's plain Run action supplies no arguments. That is intentionally
    # interpreted as the complete six-dataset experiment.
    single_mode = args.synthetic or args.csv is not None or args.download
    if args.all or not single_mode:
        train_all_downloaded(args)
    else:
        train_single(args)


if __name__ == "__main__":
    main()

