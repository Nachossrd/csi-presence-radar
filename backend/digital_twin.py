"""
Digital Twin engine for indoor localization visualization.
Manages floor plans, person positions, zones, and state broadcasting.
"""

import json
import time
import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class DigitalTwin:
    """Digital Twin representation of an indoor space."""

    def __init__(self):
        self.floor_plan: Dict[str, Any] = self.get_default_floor_plan()

        # Tracked persons: {person_id: {position, velocity, trail, zone, confidence, last_seen}}
        self.persons: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Person tracking
    # ------------------------------------------------------------------

    def update_position(self, person_id: str, position: List[float], confidence: float = 0.5):
        """Update or create a person's position in the digital twin."""
        now = time.time()

        if person_id not in self.persons:
            self.persons[person_id] = {
                "position": list(position),
                "velocity": [0.0, 0.0],
                "trail": deque(maxlen=100),
                "zone": "",
                "confidence": confidence,
                "last_seen": now,
            }
            self.persons[person_id]["trail"].append(list(position))
        else:
            person = self.persons[person_id]
            dt = now - person["last_seen"]
            dt = max(dt, 0.01)

            old_pos = person["position"]
            vx = (position[0] - old_pos[0]) / dt
            vy = (position[1] - old_pos[1]) / dt

            person["position"] = list(position)
            person["velocity"] = [round(vx, 4), round(vy, 4)]
            person["trail"].append(list(position))
            person["confidence"] = confidence
            person["last_seen"] = now

        # Detect zone
        zone = self.get_zone(position)
        self.persons[person_id]["zone"] = zone

    def get_zone(self, position: List[float]) -> str:
        """Determine which room/zone contains the given position (AABB check)."""
        x, y = position[0], position[1]

        rooms = self.floor_plan.get("rooms", [])
        for room in rooms:
            center = room.get("center", {})
            cx = center.get("x", 0)
            cz = center.get("z", 0)
            w = room.get("width", 0) / 2.0
            d = room.get("depth", 0) / 2.0

            if (cx - w) <= x <= (cx + w) and (cz - d) <= y <= (cz + d):
                return room.get("name", room.get("id", "unknown"))

        return "outside"

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Get the full JSON-serializable state for WebSocket broadcast."""
        persons_list = []
        for pid, pdata in self.persons.items():
            persons_list.append({
                "id": pid,
                "x": pdata["position"][0],
                "y": pdata["position"][1],
                "velocity": pdata["velocity"],
                "zone": pdata["zone"],
                "confidence": pdata["confidence"],
                "trail": list(pdata["trail"]),
                "last_seen": pdata["last_seen"],
            })

        return {
            "persons": persons_list,
            "floor_plan": self.floor_plan,
            "timestamp": time.time(),
        }

    def get_persons_state(self) -> Dict[str, Any]:
        """Get just the persons state (lighter payload for frequent updates)."""
        persons_list = []
        for pid, pdata in self.persons.items():
            persons_list.append({
                "id": pid,
                "x": pdata["position"][0],
                "y": pdata["position"][1],
                "velocity": pdata["velocity"],
                "zone": pdata["zone"],
                "confidence": pdata["confidence"],
                "trail": list(pdata["trail"]),
            })

        return {
            "persons": persons_list,
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # Floor plan management
    # ------------------------------------------------------------------

    def load_floor_plan(self, data: Dict[str, Any]):
        """Set floor plan from a dict."""
        self.floor_plan = data
        logger.info("Floor plan loaded with %d rooms.", len(data.get("rooms", [])))

    def save_floor_plan(self, path: str):
        """Save current floor plan to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.floor_plan, f, indent=2, ensure_ascii=False)
        logger.info("Floor plan saved to %s", path)

    def load_floor_plan_file(self, path: str) -> bool:
        """Load floor plan from JSON file."""
        p = Path(path)
        if not p.exists():
            logger.warning("Floor plan file not found: %s", path)
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.floor_plan = data
            logger.info("Floor plan loaded from %s", path)
            return True
        except Exception as exc:
            logger.error("Error loading floor plan: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Default floor plan
    # ------------------------------------------------------------------

    @staticmethod
    def get_default_floor_plan() -> Dict[str, Any]:
        """Return a realistic sample 3-bedroom house floor plan."""
        return {
            "rooms": [
                {
                    "id": "living",
                    "name": "Sala",
                    "color": "#4CAF50",
                    "walls": [
                        [0, 0, 6, 0],    # south
                        [6, 0, 6, 6],    # east
                        [6, 6, 0, 6],    # north
                        [0, 6, 0, 0],    # west
                    ],
                    "center": {"x": 3, "z": 3},
                    "width": 6,
                    "depth": 6,
                },
                {
                    "id": "kitchen",
                    "name": "Cocina",
                    "color": "#FF9800",
                    "walls": [
                        [6, 0, 10, 0],
                        [10, 0, 10, 4],
                        [10, 4, 6, 4],
                        [6, 4, 6, 0],
                    ],
                    "center": {"x": 8, "z": 2},
                    "width": 4,
                    "depth": 4,
                },
                {
                    "id": "bedroom1",
                    "name": "Habitación 1",
                    "color": "#2196F3",
                    "walls": [
                        [0, 6, 6, 6],
                        [6, 6, 6, 11],
                        [6, 11, 0, 11],
                        [0, 11, 0, 6],
                    ],
                    "center": {"x": 3, "z": 8.5},
                    "width": 6,
                    "depth": 5,
                },
                {
                    "id": "bedroom2",
                    "name": "Habitación 2",
                    "color": "#9C27B0",
                    "walls": [
                        [6, 6, 10, 6],
                        [10, 6, 10, 11],
                        [10, 11, 6, 11],
                        [6, 11, 6, 6],
                    ],
                    "center": {"x": 8, "z": 8.5},
                    "width": 4,
                    "depth": 5,
                },
                {
                    "id": "bathroom",
                    "name": "Baño",
                    "color": "#00BCD4",
                    "walls": [
                        [6, 4, 10, 4],
                        [10, 4, 10, 6],
                        [10, 6, 6, 6],
                        [6, 6, 6, 4],
                    ],
                    "center": {"x": 8, "z": 5},
                    "width": 4,
                    "depth": 2,
                },
                {
                    "id": "hallway",
                    "name": "Pasillo",
                    "color": "#607D8B",
                    "walls": [
                        [5, 3, 6, 3],
                        [6, 3, 6, 9],
                        [6, 9, 5, 9],
                        [5, 9, 5, 3],
                    ],
                    "center": {"x": 5.5, "z": 6},
                    "width": 1,
                    "depth": 6,
                },
            ],
            "dimensions": {"width": 10, "length": 11},
            "wallHeight": 2.8,
            "router": {"x": 5, "z": 3},
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    twin = DigitalTwin()

    print("=== Default Floor Plan ===")
    fp = twin.floor_plan
    print(f"Dimensions: {fp['dimensions']}")
    for room in fp["rooms"]:
        print(f"  {room['id']}: {room['name']} at ({room['center']['x']}, {room['center']['z']})")

    # Test person tracking
    twin.update_position("user1", [3.0, 3.0], 0.8)
    print(f"\nPerson at (3,3): zone = {twin.persons['user1']['zone']}")

    twin.update_position("user1", [8.0, 2.0], 0.9)
    print(f"Person at (8,2): zone = {twin.persons['user1']['zone']}")

    state = twin.get_state()
    print(f"\nState has {len(state['persons'])} persons")
