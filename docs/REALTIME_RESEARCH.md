# Real-Time Options for EmbracePlus Data

Research date: 2026-09-25. Device: Empatica EmbracePlus. Current pipeline: poll Empatica's S3 bucket (`avro_stream_poller_with_ml.py`).

## Bottom line

- **There is no publicly documented way to stream EmbracePlus data in real time.** Not over Bluetooth, and not from the cloud.
- **The S3 pipeline cannot be made near-real-time by polling faster.** The Care App uploads to the cloud in **30-minute batches**, and Empatica's data-access doc says data becomes available in the cloud **within 24 hours** of upload. Our 5 s poll interval buys almost nothing; a 30-60 s interval is enough.
- "Absolutely no way" is too strong. There are three unlocks, and all need Empatica or a hardware change (see "Options").

## What the evidence says

| Claim | Source |
|---|---|
| Data flow is EmbracePlus -> (Bluetooth) -> Care App -> (Wi-Fi/cellular) -> Empatica Cloud/S3 | Data Access PDF p.6; Empatica support |
| Care App uploads in 30-minute batches; buffers on the phone when offline | Empatica support "How the platform works" (via search summary) |
| Data accessible in the cloud within 24 h of successful upload | Data Access PDF p.6 (REV18.0) |
| Raw Avro files are created about every 30 min | Data Access PDF p.16 |
| "Live" in the Care app means live device/connection status, not live signal streaming; EmbracePlus data is "not viewable in real-time" | Empatica product/support pages (search summaries) |
| E4 had a real-time SDK and streaming server; they are retired and "can no longer collect and store data" | [E4 sunset page](https://www.empatica.com/research/e4-sunset/) |
| No Empatica device appears in the LSL supported-devices list | [LSL docs](https://labstreaminglayer.readthedocs.io/info/supported_devices.html) |
| Cloud API (Sept 2024) claims to connect data to third-party systems "in real time"; push vs pull and latency are not stated publicly; docs given only to clients | [Empatica blog](https://www.empatica.com/blog/cloud-api-revolutionizing-data-integration-for-clinical-trials/) |

## How the bucket credentials work

- Credentials are not AWS IAM users you own. They are **Data Access Keys** issued by Empatica through the **Care Portal**: *Data -> Data Access Key -> Generate Access Key*.
- You get a CSV with `AWS Access Key ID`, `AWS Secret Access Key`, and an `S3 Access URL` (bucket + prefix for your org/study/site). **The secret is shown once.** Keys can be revoked and regenerated there.
- Care Portal access comes from your Empatica account manager.
- The key is scoped to Empatica's bucket, which is why we cannot add S3 event notifications ourselves.
- **Correction to my earlier advice:** the exposed key in this repo should be revoked and regenerated in the **Care Portal**, not in AWS IAM. It has been in git history since commit `b139263`.

## How other people do it

Every public tool I found is batch/post-hoc, none is real-time:

- Empatica's own Avro-to-CSV Python script (in the Data Access PDF appendix).
- [empaticaR](https://github.com/brian-lau/empaticaR): R package reading the S3 bucket.
- [LOTUS](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11644628/): MATLAB toolkit for stitching 30-minute Avro chunks.
- [EmbracePlus-Visualizer](https://github.com/JocelynVelarde/EmbracePlus-Visualizer) and [Physio_processing](https://github.com/abequet/Physio_processing): Avro readers.

I found no public reverse-engineering of the EmbracePlus Bluetooth protocol, and no open-source real-time EmbracePlus pipeline.

## Options, ranked

### 1. Ask Empatica (do this first, it is free)
Questions for the account manager:
1. Does the **Cloud API** push (webhook) or pull? What latency is guaranteed? Which plan includes it? Does it expose per-minute biomarkers (EDA, temperature, pulse rate) or only participant/adherence data?
2. Can Empatica enable **cross-account S3 event notifications** (EventBridge/SQS) from the bucket to our AWS account?
3. Is there any **enterprise/research real-time or Bluetooth access** for EmbracePlus (SDK, streaming mode, shorter upload interval)?
4. Can the Care App upload interval be shortened for a study?

### 2. Keep S3, be honest about latency
- Realistic freshness is **30 min to a few hours**, and worst case is 24 h. Fine for post-session or "recent history" views, not for live feedback.
- Change `POLL_SECONDS` to 30-60. Use `observed_latency_seconds` (already in `/latest`) to measure the real lag and put that number in front of Empatica.

### 3. Add a second, truly real-time sensor for the live view
| Device | Real-time path | Signals | Caveat |
|---|---|---|---|
| Shimmer3 GSR+ | Bluetooth streaming, Consensys, LSL examples | EDA (64 Hz), PPG, IMU | Closest E4 replacement for EDA; wired/wrist form factor differs |
| BITalino | LSL (Python) | EDA, ECG, etc. | Research kit, less wearable |
| Polar Verity Sense / H10 | Official BLE SDK | HR, IBI, PPG, ECG, motion | **No EDA or skin temperature** |

Our models use **EDA + temperature + minute index**, so a live view needs an EDA source (Shimmer). Different sensors give different signal characteristics, so the models would need re-validation or retraining. Polar alone cannot feed them.

### 4. Hybrid (probably the practical answer)
Live dashboard from a real-time sensor; EmbracePlus S3 data for validated batch analysis and model training.

### 5. Unofficial Bluetooth reverse engineering: not recommended
No public prior work, likely breaks the pairing/encryption with the Care App, may violate Empatica's terms, and this is a medical-grade research device. Not worth the risk.

## Limits of this research
- Several Empatica support pages returned 403, so some facts (30-minute batch, "not viewable in real-time") come from search-result summaries, not the pages themselves. Confirm with Empatica.
- Cloud API documentation is client-only; its real capabilities are unverified.
- I could not open the Springer paper on real-time emotion recognition, so I don't know what device or architecture it used.
