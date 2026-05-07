import csv
import functools
import fcntl
import os
import time
from datetime import datetime, timezone

from codecarbon import EmissionsTracker


def _empty_metrics() -> dict:
    metrics = {
        "energy_consumed_kwh": "",
        "cpu_energy_kwh": "",
        "gpu_energy_kwh": "",
        "ram_energy_kwh": "",
        "cpu_power_watts": "",
        "gpu_power_watts": "",
        "ram_power_watts": "",
    }
    return metrics


def _find_value(row: dict, candidates: list[str]):
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for name in candidates:
        value = lowered.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return ""


def _raw_metrics_dir(report_path: str) -> str:
    return os.path.join(os.path.dirname(report_path), "raw")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _raw_metrics_filename(func_name: str) -> str:
    pod = _safe_name(_pod_name())
    func = _safe_name(func_name)
    ts = int(time.time() * 1000)
    return f"{pod}-{func}-{ts}.csv"


def _load_metrics_from_raw_csv(raw_csv_path: str) -> dict:
    metrics = _empty_metrics()
    if not os.path.exists(raw_csv_path):
        return metrics

    rows = []
    with open(raw_csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return metrics

    row = rows[-1]

    # Column aliases cover CodeCarbon naming differences across versions.
    metrics["energy_consumed_kwh"] = f"{_safe_float(_find_value(row, ['energy_consumed', 'energy_consumed_kwh'])):.12f}" if _find_value(row, ['energy_consumed', 'energy_consumed_kwh']) != "" else ""
    metrics["cpu_energy_kwh"] = f"{_safe_float(_find_value(row, ['cpu_energy', 'cpu_energy_kwh'])):.12f}" if _find_value(row, ['cpu_energy', 'cpu_energy_kwh']) != "" else ""
    metrics["gpu_energy_kwh"] = f"{_safe_float(_find_value(row, ['gpu_energy', 'gpu_energy_kwh'])):.12f}" if _find_value(row, ['gpu_energy', 'gpu_energy_kwh']) != "" else ""
    metrics["ram_energy_kwh"] = f"{_safe_float(_find_value(row, ['ram_energy', 'ram_energy_kwh'])):.12f}" if _find_value(row, ['ram_energy', 'ram_energy_kwh']) != "" else ""
    metrics["cpu_power_watts"] = f"{_safe_float(_find_value(row, ['cpu_power', 'cpu_power_watts'])):.12f}" if _find_value(row, ['cpu_power', 'cpu_power_watts']) != "" else ""
    metrics["gpu_power_watts"] = f"{_safe_float(_find_value(row, ['gpu_power', 'gpu_power_watts'])):.12f}" if _find_value(row, ['gpu_power', 'gpu_power_watts']) != "" else ""
    metrics["ram_power_watts"] = f"{_safe_float(_find_value(row, ['ram_power', 'ram_power_watts'])):.12f}" if _find_value(row, ['ram_power', 'ram_power_watts']) != "" else ""

    return metrics


def _report_path() -> str:
    report_path = os.environ.get("CODECARBON_REPORT_PATH", "").strip()
    if report_path:
        return report_path

    workflow_name = os.environ.get("WORKFLOW_NAME", "").strip()
    workflow_dir = workflow_name or "current"
    return os.path.join("/app/reports/codecarbon", workflow_dir, "emissions.csv")


def _lock_path(report_path: str) -> str:
    return f"{report_path}.lock"


def _pod_name() -> str:
    return (os.environ.get("HOSTNAME") or os.environ.get("POD_NAME") or "unknown").strip() or "unknown"


def _workflow_name() -> str:
    return (os.environ.get("WORKFLOW_NAME") or os.environ.get("ARGO_WORKFLOW_NAME") or "").strip()


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _write_emission_summary(record: dict) -> None:
    report_path = _report_path()
    directory = os.path.dirname(report_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    lock_path = _lock_path(report_path)
    with open(lock_path, "a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        rows = []

        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8", newline="") as existing_file:
                reader = csv.DictReader(existing_file)
                rows = [row for row in reader if row.get("scope") == "pod"]

        rows = [
            row
            for row in rows
            if not (
                row.get("workflow_name") == record["workflow_name"]
                and row.get("pod_name") == record["pod_name"]
                and row.get("function_name") == record["function_name"]
            )
        ]
        rows.append(record)

        # Aggregate metrics across pods for each numeric column
        total_emissions = sum(_safe_float(row.get("emissions_kg_co2eq", 0.0)) for row in rows)
        total_energy = sum(_safe_float(row.get("energy_consumed_kwh", 0.0)) for row in rows)
        total_cpu_energy = sum(_safe_float(row.get("cpu_energy_kwh", 0.0)) for row in rows)
        total_gpu_energy = sum(_safe_float(row.get("gpu_energy_kwh", 0.0)) for row in rows)
        total_ram_energy = sum(_safe_float(row.get("ram_energy_kwh", 0.0)) for row in rows)
        
        summary_rows = rows + [{
            "scope": "total",
            "workflow_name": record["workflow_name"],
            "pod_name": "ALL_PODS",
            "function_name": "ALL_FUNCTIONS",
            "emissions_kg_co2eq": f"{total_emissions:.12f}",
            "energy_consumed_kwh": f"{total_energy:.12f}",
            "cpu_energy_kwh": f"{total_cpu_energy:.12f}",
            "gpu_energy_kwh": f"{total_gpu_energy:.12f}",
            "ram_energy_kwh": f"{total_ram_energy:.12f}",
            "cpu_power_watts": "",
            "gpu_power_watts": "",
            "ram_power_watts": "",
            "started_at": "",
            "finished_at": "",
        }]

        fieldnames = [
            "scope",
            "workflow_name",
            "pod_name",
            "function_name",
            "emissions_kg_co2eq",
            "energy_consumed_kwh",
            "cpu_energy_kwh",
            "gpu_energy_kwh",
            "ram_energy_kwh",
            "cpu_power_watts",
            "gpu_power_watts",
            "ram_power_watts",
            "started_at",
            "finished_at",
        ]
        temp_path = f"{report_path}.tmp"
        with open(temp_path, "w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

        os.replace(temp_path, report_path)


def ccdecorator(func):
    @functools.wraps(func)
    def _wrapped(*args, **kwargs):
        report_path = _report_path()
        raw_dir = _raw_metrics_dir(report_path)
        os.makedirs(raw_dir, exist_ok=True)
        raw_file = _raw_metrics_filename(func.__name__)

        tracker = EmissionsTracker(
            project_name=func.__name__,
            save_to_file=True,
            output_dir=raw_dir,
            output_file=raw_file,
        )
        started_at = datetime.now(timezone.utc).isoformat()
        tracker.start()
        result = None
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            try:
                emissions = tracker.stop()
                finished_at = datetime.now(timezone.utc).isoformat()
                record = {
                    "scope": "pod",
                    "workflow_name": _workflow_name(),
                    "pod_name": _pod_name(),
                    "function_name": func.__name__,
                    "emissions_kg_co2eq": f"{_safe_float(emissions):.12f}",
                    "started_at": started_at,
                    "finished_at": finished_at,
                }
                record.update(_load_metrics_from_raw_csv(os.path.join(raw_dir, raw_file)))

                _write_emission_summary(record)
                print(
                    f"CodeCarbon emissions for {func.__name__} on {_pod_name()}: "
                    f"{record['emissions_kg_co2eq']} kg CO2eq"
                )
                print(f"CodeCarbon summary written to {_report_path()}")
            except Exception:
                pass

        return result

    return _wrapped