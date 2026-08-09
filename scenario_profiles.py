"""Validated, user-editable mission scenario profiles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


ALLOWED_NAVIGATION_POLICIES = frozenset(
    ("patrol", "rescue", "target", "lego", "waypoint", "return_home", "stationary")
)
ALLOWED_COMPLETION_CONDITIONS = frozenset(
    ("operator", "waypoint_reached", "home_reached")
)


@dataclass(frozen=True)
class ScenarioProfile:
    id: str
    name: str
    short_name: str
    objective: str
    difficulty: str
    system_prompt: str
    scene_prompt: str
    navigation_policy: str
    key: str | None = None
    preferred_target: str | None = None
    target_labels: tuple = ()
    use_lego_vision: bool = False
    allow_motion: bool = True
    requires_cloud: bool = False
    requires_microphone: bool = False
    completion: str = "operator"

    @classmethod
    def from_dict(cls, data):
        required = (
            "id",
            "name",
            "short_name",
            "objective",
            "difficulty",
            "system_prompt",
            "scene_prompt",
            "navigation_policy",
        )
        missing = [field for field in required if not str(data.get(field, "")).strip()]
        if missing:
            raise ValueError("Scenario is missing required fields: " + ", ".join(missing))
        policy = data["navigation_policy"]
        if policy not in ALLOWED_NAVIGATION_POLICIES:
            raise ValueError(f"Unsupported navigation policy: {policy}")
        key = data.get("key")
        if key is not None and (not isinstance(key, str) or len(key) != 1):
            raise ValueError("Scenario key must be one character or null.")
        raw_target_labels = data.get("target_labels", ())
        if isinstance(raw_target_labels, (str, bytes)):
            raise ValueError("Scenario target_labels must be an array.")
        target_labels = tuple(str(label).strip() for label in raw_target_labels)
        if any(not label for label in target_labels):
            raise ValueError("Scenario target labels cannot be empty.")
        if len(target_labels) != len(set(target_labels)):
            raise ValueError("Scenario target labels must be unique.")
        preferred_target = data.get("preferred_target")
        if target_labels and preferred_target and preferred_target not in target_labels:
            raise ValueError("Scenario preferred target must be in target_labels.")
        completion = data.get("completion", "operator")
        if completion not in ALLOWED_COMPLETION_CONDITIONS:
            raise ValueError(f"Unsupported completion condition: {completion}")
        completion_policies = {
            "waypoint_reached": "waypoint",
            "home_reached": "return_home",
        }
        required_policy = completion_policies.get(completion)
        if required_policy and policy != required_policy:
            raise ValueError(
                f"Completion condition {completion} requires {required_policy} navigation."
            )
        return cls(
            id=data["id"],
            name=data["name"],
            short_name=data["short_name"],
            objective=data["objective"],
            difficulty=data["difficulty"],
            system_prompt=data["system_prompt"],
            scene_prompt=data["scene_prompt"],
            navigation_policy=policy,
            key=key,
            preferred_target=preferred_target,
            target_labels=target_labels,
            use_lego_vision=bool(data.get("use_lego_vision", False)),
            allow_motion=bool(data.get("allow_motion", True)),
            requires_cloud=bool(data.get("requires_cloud", False)),
            requires_microphone=bool(data.get("requires_microphone", False)),
            completion=completion,
        )

    def accepts_target(self, label):
        return bool(label) and (not self.target_labels or label in self.target_labels)


class ScenarioCatalog:
    def __init__(self, profiles):
        profiles = tuple(profiles)
        if not profiles:
            raise ValueError("At least one scenario profile is required.")
        ids = [profile.id for profile in profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("Scenario IDs must be unique.")
        keys = [profile.key.lower() for profile in profiles if profile.key]
        if len(keys) != len(set(keys)):
            raise ValueError("Scenario keyboard shortcuts must be unique.")
        self._profiles = profiles
        self._by_id = {profile.id: profile for profile in profiles}

    @classmethod
    def load(cls, path):
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("Unsupported scenarios.json version; expected 1.")
        return cls(ScenarioProfile.from_dict(item) for item in payload["scenarios"])

    def __iter__(self):
        return iter(self._profiles)

    def __len__(self):
        return len(self._profiles)

    def __contains__(self, scenario_id):
        return scenario_id in self._by_id

    def get(self, scenario_id):
        try:
            return self._by_id[scenario_id]
        except KeyError as error:
            raise KeyError(f"Unknown scenario: {scenario_id}") from error

    @property
    def default(self):
        return self._profiles[0]

    @property
    def key_map(self):
        key_map = {}
        for profile in self._profiles:
            if not profile.key:
                continue
            for variant in {profile.key.lower(), profile.key.upper()}:
                key_map[ord(variant)] = profile.id
        return key_map

    def as_legacy_dict(self):
        return {
            profile.id: {
                "name": profile.name,
                "system_prompt": profile.system_prompt,
                "scene_prompt": profile.scene_prompt,
            }
            for profile in self._profiles
        }
