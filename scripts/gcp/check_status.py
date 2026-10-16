import subprocess
import json

filter_expr = 'resource.type="cloud_run_job" AND labels."run.googleapis.com/execution_name"="voynich-batch-job-d2vpr"'
cmd = [
    "gcloud", "logging", "read",
    filter_expr,
    "--project=ether-509812",
    "--billing-project=ether-509812",
    "--limit=25",
    "--format=json"
]

res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
if res.returncode != 0:
    print("Error:", res.stderr)
else:
    try:
        logs = json.loads(res.stdout)
        print(f"Total entries: {len(logs)}")
        for e in reversed(logs):
            ts = e.get("timestamp", "")
            msg = e.get("textPayload") or e.get("jsonPayload", {}).get("message")
            if msg:
                print(f"[{ts}] {msg}")
    except Exception as ex:
        print("JSON parse error:", ex, res.stdout[:200])
