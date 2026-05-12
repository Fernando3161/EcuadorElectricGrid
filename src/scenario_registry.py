from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    year: int
    title: str
    description: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    tags: List[str]
    active: bool = True

    def resolve_paths(self, project_root: Path) -> Dict[str, Dict[str, Path]]:
        in_paths = {k: (project_root / v).resolve() for k, v in self.inputs.items()}
        out_paths = {k: (project_root / v).resolve() for k, v in self.outputs.items()}
        return {"inputs": in_paths, "outputs": out_paths}


class ScenarioRegistry:
    def __init__(self, registry_path: Path, project_root: Optional[Path] = None) -> None:
        self.registry_path = Path(registry_path)
        self.project_root = project_root or self._infer_project_root()

        self._raw: Dict[str, Any] = {}
        self._scenarios: Dict[str, Scenario] = {}

    def _infer_project_root(self) -> Path:
        # Assumption: registry is in <repo>/src/scenario_registry.json
        # so project root is parent of src
        rp = self.registry_path.resolve()
        if rp.parent.name == "src":
            return rp.parent.parent
        return Path.cwd().resolve()

    def load(self) -> "ScenarioRegistry":
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry file not found at {self.registry_path}")

        with self.registry_path.open("r", encoding="utf-8") as f:
            self._raw = json.load(f)

        self._scenarios = self._parse(self._raw)
        return self

    @staticmethod
    def _parse(raw: Dict[str, Any]) -> Dict[str, Scenario]:
        if "scenarios" not in raw or not isinstance(raw["scenarios"], dict):
            raise ValueError("Invalid registry format: missing scenarios mapping")

        scenarios: Dict[str, Scenario] = {}
        for scenario_id, cfg in raw["scenarios"].items():
            scenarios[scenario_id] = Scenario(
                scenario_id=scenario_id,
                year=int(cfg["year"]),
                title=str(cfg.get("title", "")),
                description=str(cfg.get("description", "")),
                inputs=dict(cfg.get("inputs", {})),
                outputs=dict(cfg.get("outputs", {})),
                tags=list(cfg.get("tags", [])),
                active=bool(cfg.get("active", True)),
            )
        return scenarios

    def list_ids(self, only_active: bool = True) -> List[str]:
        if only_active:
            return [sid for sid, s in self._scenarios.items() if s.active]
        return list(self._scenarios.keys())

    def get(self, scenario_id: str) -> Scenario:
        if scenario_id not in self._scenarios:
            raise KeyError(f"Scenario not found: {scenario_id}")
        return self._scenarios[scenario_id]

    def validate_inputs(self, scenario_id: str, strict: bool = False) -> List[Path]:
        scenario = self.get(scenario_id)
        paths = scenario.resolve_paths(self.project_root)["inputs"]
        missing = [p for p in paths.values() if not p.exists()]

        if strict and missing:
            missing_str = "\n".join(str(p) for p in missing)
            raise FileNotFoundError(f"Missing input files:\n{missing_str}")

        return missing
