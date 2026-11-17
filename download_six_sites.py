"""Download the six representative NSRDB datasets from inside PyCharm.

Fill local_credentials.py first, then open this file in PyCharm and click Run.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

from data import download_nsrdb_csv, read_nsrdb_ghi, select_paper_window
from local_credentials import NSRDB_API_KEY, NSRDB_EMAIL


# Reproducible replacement sites. The paper does not publish its coordinates.
SITES = (
    {"dataset": 1, "country": "Cuba", "city": "Havana", "latitude": 23.1136, "longitude": -82.3666},
    {"dataset": 2, "country": "Mexico", "city": "Mexico City", "latitude": 19.4326, "longitude": -99.1332},
    {"dataset": 3, "country": "Bolivia", "city": "Santa Cruz", "latitude": -17.7833, "longitude": -63.1821},
    {"dataset": 4, "country": "Suriname", "city": "Paramaribo", "latitude": 5.8520, "longitude": -55.2038},
    {"dataset": 5, "country": "Venezuela", "city": "Caracas", "latitude": 10.4806, "longitude": -66.9036},
    {"dataset": 6, "country": "United States", "city": "Phoenix", "latitude": 33.4484, "longitude": -112.0740},
)

YEAR = 2022
WINDOW_START = date(2022, 1, 1)
OVERWRITE_EXISTING = False
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "data" / f"nsrdb_{YEAR}"


def safe_name(city: str) -> str:
    return city.lower().replace(" ", "_")


def get_credentials() -> tuple[str, str]:
    api_key = NSRDB_API_KEY.strip()
    email = NSRDB_EMAIL.strip()

    if not api_key:
        raise ValueError("请先在 local_credentials.py 中填写 API Key。")
    if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        raise ValueError("请先在 local_credentials.py 中填写有效邮箱。")
    return api_key, email


def validate_download(path: Path) -> tuple[int, int]:
    records = read_nsrdb_ghi(path)
    expected_year_records = 8784 if YEAR % 4 == 0 else 8760
    if len(records) != expected_year_records:
        raise ValueError(
            f"全年记录数应为 {expected_year_records}，实际为 {len(records)}。"
        )

    paper_window = select_paper_window(records, start_date=WINDOW_START)
    if len(paper_window) != 1708:
        raise ValueError(f"论文式122天窗口应为1708点，实际为 {len(paper_window)}。")
    return len(records), len(paper_window)


def main() -> None:
    print("DAN/NSRDB 六城市下载器")
    print(f"年份: {YEAR}; 122天窗口起点: {WINDOW_START.isoformat()}")
    print(f"保存目录: {OUTPUT_DIR}")
    print()

    api_key, email = get_credentials()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    failures: list[str] = []

    for index, site in enumerate(SITES, start=1):
        city_file = safe_name(site["city"])
        destination = OUTPUT_DIR / (
            f"dataset{site['dataset']}_{city_file}_{YEAR}.csv"
        )
        print(
            f"[{index}/{len(SITES)}] Dataset{site['dataset']} "
            f"{site['city']}, {site['country']}"
        )

        try:
            reused = False
            if destination.exists() and not OVERWRITE_EXISTING:
                print("  文件已存在，正在验证。")
                try:
                    year_records, window_points = validate_download(destination)
                    reused = True
                    print("  现有文件有效，直接复用。")
                except Exception as validation_error:
                    print(f"  现有文件无效，将重新下载：{validation_error}")
                    destination.unlink()

            if not reused:
                print("  正在从 NSRDB 下载……", flush=True)
                download_nsrdb_csv(
                    destination,
                    api_key=api_key,
                    email=email,
                    latitude=site["latitude"],
                    longitude=site["longitude"],
                    year=YEAR,
                )
                year_records, window_points = validate_download(destination)

            print(
                f"  完成：{destination.name}; "
                f"全年 {year_records} 条，论文窗口 {window_points} 点。"
            )
            manifest.append(
                {
                    **site,
                    "year": YEAR,
                    "window_start": WINDOW_START.isoformat(),
                    "window_days": 122,
                    "year_records": year_records,
                    "paper_window_points": window_points,
                    "file": destination.name,
                }
            )
        except Exception as exc:  # Keep trying the remaining independent sites.
            if destination.exists() and destination.stat().st_size < 1000:
                destination.unlink()
            message = f"Dataset{site['dataset']} {site['city']}: {exc}"
            failures.append(message)
            print(f"  失败：{exc}", file=sys.stderr)

        # Direct CSV requests are limited to at most one request per second.
        if index < len(SITES):
            time.sleep(1.1)

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print(f"成功下载/验证：{len(manifest)}/{len(SITES)}")
    print(f"清单：{manifest_path}")
    if failures:
        print("失败项目：", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)

    print("六个数据集均已准备完毕。")


if __name__ == "__main__":
    main()

