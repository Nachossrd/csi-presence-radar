"""Fire fake radar events at the dashboard for end-to-end testing.

Run the FastAPI server first:
    uvicorn backend.api:app --reload

Then in another terminal:
    python -m backend.test_radar_event burst                # 1 motion + heatmap
    python -m backend.test_radar_event loop --interval 3    # forever
    python -m backend.test_radar_event tag --name "Bano"    # simulate a tag firing
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8000"
ANCHORS = ["starlink_hotspot", "tv_living", "smartwatch_yo"]


def _post(url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/api/radar/event", data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            body = json.loads(resp.read())
            print(f"  -> {payload['kind']:8s} {payload.get('anchor_id', ''):20s} "
                  f"broadcast_to={body.get('broadcast_to')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [error] {exc}")


def cmd_burst(args) -> None:
    """Fire 1 motion event on a random anchor."""
    anchor = random.choice(ANCHORS)
    _post(args.url, {
        "anchor_id": anchor,
        "source": "test",
        "kind": "motion",
        "state": "motion",
        "value": round(random.uniform(0.3, 0.9), 2),
    })


def cmd_loop(args) -> None:
    """Fire random events forever."""
    print(f"Looping events every {args.interval}s. Ctrl+C to stop.")
    try:
        while True:
            anchor = random.choice(ANCHORS)
            kind = random.choice(["state", "state", "motion"])
            state = random.choice(["quiet", "quiet", "quiet", "motion", "moving"])
            _post(args.url, {
                "anchor_id": anchor, "source": "test",
                "kind": kind, "state": state,
                "value": round(random.uniform(0.0, 1.0), 2),
            })
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


def cmd_tag(args) -> None:
    """Simulate a tag firing (direction matched)."""
    _post(args.url, {
        "source": "tag", "kind": "tag_fired", "state": "motion",
        "extra": {"name": args.name},
    })


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=DEFAULT_URL)
    sub = p.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("burst", help="1 motion event")
    pb.set_defaults(func=cmd_burst)
    pl = sub.add_parser("loop", help="Fire random events forever")
    pl.add_argument("--interval", type=float, default=3.0)
    pl.set_defaults(func=cmd_loop)
    pt = sub.add_parser("tag", help="Simulate a tag firing")
    pt.add_argument("--name", default="Bano")
    pt.set_defaults(func=cmd_tag)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
