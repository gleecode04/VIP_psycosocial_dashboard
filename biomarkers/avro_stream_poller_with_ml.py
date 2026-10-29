import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta

import boto3
import pandas as pd
from flask import Flask, jsonify

from biomarker_runtime import LiveBiomarkerPredictor
from risk_score import RollingRiskScorer, band_for
from system_message_api import register_system_message_routes

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

S3_BUCKET = os.environ.get("S3_BUCKET", "empatica-us-east-1-prod-data")
S3_BASE_PREFIX = os.environ.get("S3_BASE_PREFIX", "v2/1324/1/1/participant_data/")

# S3 list calls are cheap; the real latency floor is how often Empatica's
# Care App itself uploads new objects to S3, not this interval. There is no
# push/webhook option available against a bucket we don't own, so tight
# polling is the closest thing to "real time" achievable from our side.
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "5"))

# How long after UTC midnight to keep checking yesterday's prefix too, in
# case in-flight uploads still land under the previous day's date folder.
MIDNIGHT_GRACE_MINUTES = int(os.environ.get("MIDNIGHT_GRACE_MINUTES", "15"))

FORCE_DOWNLOAD_EVERY_POLL = False

LOCAL_BIOMARKER_DIR = r"bio"
os.makedirs(LOCAL_BIOMARKER_DIR, exist_ok=True)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
predictor = LiveBiomarkerPredictor(MODEL_DIR)
risk_scorer = RollingRiskScorer()

app = Flask(__name__)
latest_sample = {
    "eda": None,
    "temp": None,
    "pulse_rate": None,
    "eda_time": None,
    "temp_time": None,
    "pulse_time": None,
    "participant_id": None,
    "source_prefix": None,
    "updated_at": None,
    "observed_latency_seconds": None,
    "note": "waiting for first sample",
    "predictions": None,
}


@app.route("/latest", methods=["GET"])
def latest():
    return jsonify(latest_sample)


register_system_message_routes(app, latest_sample)


def make_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def list_objects_all(s3, prefix):
    out = []
    token = None
    while True:
        args = {"Bucket": S3_BUCKET, "Prefix": prefix}
        if token:
            args["ContinuationToken"] = token
        resp = s3.list_objects_v2(**args)
        out.extend(resp.get("Contents", []))
        if resp.get("IsTruncated"):
            token = resp["NextContinuationToken"]
        else:
            break
    return out


def download_key(s3, key, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(S3_BUCKET, key, local_path)


def looks_like_epoch_ms(x):
    try:
        return float(x) > 1e11
    except Exception:
        return False


def epoch_ms_to_iso(v):
    try:
        dt = datetime.fromtimestamp(float(v) / 1000.0, tz=timezone.utc).astimezone()
        return dt.isoformat(timespec="seconds")
    except Exception:
        return None


def find_timestamp_col(df):
    for c in df.columns:
        cl = c.lower()
        if "timestamp_iso" == cl:
            return c
    for c in df.columns:
        cl = c.lower()
        if "timestamp_unix" == cl:
            return c
    for c in df.columns:
        cl = c.lower()
        if "time" in cl or "stamp" in cl or "unix" in cl or "epoch" in cl:
            return c
    return None


def choose_value_col(df, kind):
    preferred = {
        "eda": ["eda_scl_usiemens", "eda", "EDA", "value"],
        "temp": ["temperature_celsius", "temp_celsius", "temperature", "temp", "value"],
        "pulse": ["pulse_rate", "pulse-rate", "pulse", "hr", "heart_rate", "bpm", "value"],
    }[kind]

    for c in preferred:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c

    ts_col = find_timestamp_col(df)
    for c in df.columns:
        if c != ts_col and pd.api.types.is_numeric_dtype(df[c]):
            try:
                if not looks_like_epoch_ms(df.iloc[-1][c]):
                    return c
            except Exception:
                return c
    return None


def read_last_value(csv_path, kind):
    df = pd.read_csv(csv_path)
    if df.empty:
        return None, None

    ts_col = find_timestamp_col(df)
    val_col = choose_value_col(df, kind)
    if not val_col:
        print(f"[DEBUG] No value column for {kind}: {list(df.columns)}")
        return None, None

    for i in range(len(df) - 1, -1, -1):
        row = df.iloc[i]
        val = row.get(val_col, None)
        if pd.isna(val) or looks_like_epoch_ms(val):
            continue

        ts_val = None
        if ts_col and ts_col in df.columns:
            ts_raw = row.get(ts_col, None)
            if pd.notna(ts_raw):
                if looks_like_epoch_ms(ts_raw):
                    ts_val = epoch_ms_to_iso(ts_raw)
                else:
                    ts_val = str(ts_raw)

        return ts_val, float(val)

    return None, None


def active_date_strings(now_utc):
    """Only look back at yesterday during a short grace window after
    midnight UTC, instead of scanning two full days on every single poll."""
    today = now_utc.strftime("%Y-%m-%d")
    minutes_since_midnight = now_utc.hour * 60 + now_utc.minute
    if minutes_since_midnight <= MIDNIGHT_GRACE_MINUTES:
        yesterday = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
        return [today, yesterday]
    return [today]


def pick_most_recent_participant_and_keys(s3, date_prefixes, date_strings):
    candidates = []

    with ThreadPoolExecutor(max_workers=max(1, len(date_prefixes))) as pool:
        listings = pool.map(lambda p: list_objects_all(s3, p), date_prefixes)

    for day, objs in zip(date_strings, listings):
        for o in objs:
            k = o["Key"]
            if "/digital_biomarkers/aggregated_per_minute/" not in k:
                continue
            if not k.lower().endswith(".csv"):
                continue
            if f"_{day}_" not in k:
                continue
            candidates.append(o)

    if not candidates:
        return None, {}, {}, None

    groups = {}
    for o in candidates:
        parts = o["Key"].split("/")
        pid = None
        for day in date_strings:
            if day in parts:
                try:
                    pid = parts[parts.index(day) + 1]
                except Exception:
                    pid = None
                if pid:
                    break
        if not pid:
            continue
        groups.setdefault(pid, []).append(o)

    if not groups:
        return None, {}, {}, None

    best_pid = max(groups, key=lambda p: max(x["LastModified"] for x in groups[p]))
    keys = {}
    mods = {}
    used_prefix = None
    best_objs = groups[best_pid]

    def newest_ending(suffix):
        matches = [o for o in best_objs if o["Key"].endswith(suffix)]
        if not matches:
            return None
        return max(matches, key=lambda x: x["LastModified"])

    eda_o = newest_ending("_eda.csv")
    temp_o = newest_ending("_temperature.csv")
    pulse_o = newest_ending("_pulse-rate.csv")

    if eda_o:
        keys["eda"] = eda_o["Key"]
        mods["eda"] = eda_o["LastModified"]
    if temp_o:
        keys["temp"] = temp_o["Key"]
        mods["temp"] = temp_o["LastModified"]
    if pulse_o:
        keys["pulse"] = pulse_o["Key"]
        mods["pulse"] = pulse_o["LastModified"]

    newest_obj = max(best_objs, key=lambda x: x["LastModified"])
    newest_key_parts = newest_obj["Key"].split("/")
    for day in date_strings:
        if day in newest_key_parts:
            used_prefix = f"{S3_BASE_PREFIX}{day}/"
            break

    return best_pid, keys, mods, used_prefix


def poll_loop():
    s3 = make_s3_client()
    last_sig = None
    last_mods = {}

    print(f"Polling biomarker CSV updates every {POLL_SECONDS}s...")

    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            days = active_date_strings(now_utc)
            prefixes = [f"{S3_BASE_PREFIX}{d}/" for d in days]
            pid, keys, mods, used_prefix = pick_most_recent_participant_and_keys(s3, prefixes, days)

            if not pid or not all(k in keys for k in ("eda", "temp", "pulse")):
                latest_sample["note"] = "waiting for biomarker files"
                time.sleep(POLL_SECONDS)
                continue

            changed = FORCE_DOWNLOAD_EVERY_POLL or any(mods[k] != last_mods.get(k) for k in mods)

            print(f"[DEBUG] Participant: {pid}")
            print(f"[DEBUG] S3 LastModified eda={mods.get('eda')} temp={mods.get('temp')} pulse={mods.get('pulse')}")
            print(f"[DEBUG] changed={changed} (force={FORCE_DOWNLOAD_EVERY_POLL})")

            if not changed:
                time.sleep(POLL_SECONDS)
                continue

            local_day = used_prefix.split("/")[-2] if used_prefix else days[0]
            local_dir = os.path.join(LOCAL_BIOMARKER_DIR, local_day, pid)
            paths = {
                "eda": os.path.join(local_dir, "eda.csv"),
                "temp": os.path.join(local_dir, "temperature.csv"),
                "pulse": os.path.join(local_dir, "pulse-rate.csv"),
            }

            with ThreadPoolExecutor(max_workers=len(paths)) as pool:
                list(pool.map(lambda k: download_key(s3, keys[k], paths[k]), paths))
            for k in paths:
                last_mods[k] = mods[k]

            eda_ts, eda_val = read_last_value(paths["eda"], "eda")
            temp_ts, temp_val = read_last_value(paths["temp"], "temp")
            pulse_ts, pulse_val = read_last_value(paths["pulse"], "pulse")

            sig = (eda_ts, temp_ts, pulse_ts, eda_val, temp_val, pulse_val)

            if sig == last_sig:
                latest_sample["note"] = "S3 updated but last valid value unchanged (no new minute row yet)"
                print("[DEBUG] last valid value unchanged:", sig)
            else:
                newest_upload = max(mods.values())
                observed_latency = (datetime.now(timezone.utc) - newest_upload).total_seconds()

                latest_sample.update(
                    {
                        "eda": eda_val,
                        "temp": temp_val,
                        "pulse_rate": pulse_val,
                        "eda_time": eda_ts,
                        "temp_time": temp_ts,
                        "pulse_time": pulse_ts,
                        "participant_id": pid,
                        "source_prefix": used_prefix if used_prefix else prefixes[0],
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "observed_latency_seconds": round(observed_latency, 1),
                        "note": "new minute detected",
                    }
                )
                latest_sample["predictions"] = predictor.update(latest_sample)
                predictions = latest_sample["predictions"]
                frustration_info = predictions.get("frustration") or {}
                mental_info = predictions.get("mental_demand") or {}
                predictions["risk"] = risk_scorer.score(
                    frustration_label=frustration_info.get("label"),
                    mental_label=mental_info.get("label"),
                    frustration_confidence=frustration_info.get("confidence"),
                    mental_confidence=mental_info.get("confidence"),
                )
                last_sig = sig
                print(f"Updated /latest (observed S3 latency: {observed_latency:.1f}s):", latest_sample)

                overall = predictions["risk"]["overall"]
                print(
                    f"[ML] {latest_sample['updated_at']} participant={pid} "
                    f"eda={eda_val} temp={temp_val} pulse={pulse_val} "
                    f"-> prediction: {band_for(overall).upper()} "
                    f"(frustration={frustration_info.get('label')} {frustration_info.get('confidence')}, "
                    f"mental_demand={mental_info.get('label')} {mental_info.get('confidence')}, "
                    f"risk={overall})"
                )

        except Exception as e:
            print("Poll error:", e)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    import threading

    threading.Thread(target=poll_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=7001)
