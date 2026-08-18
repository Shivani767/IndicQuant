"""Config loading and provenance.

Every artifact in this project is addressed by a config hash so a stage is never recomputed.
Re-quantizing a 32B model is the dominant cost of the whole project, so the caching contract
is load-bearing rather than a convenience.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"


class ConfigError(RuntimeError):
    """Raised when a config is missing, malformed, or internally inconsistent."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = CONFIG_ROOT / path
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"config must be a mapping, got {type(data).__name__}: {path}")
    return data


def load_model_config(name: str) -> dict[str, Any]:
    return load_yaml(Path("model") / f"{name}.yaml")


def load_calibration_config(name: str) -> dict[str, Any]:
    return load_yaml(Path("calibration") / f"{name}.yaml")


def load_condition_config(condition_id: str) -> dict[str, Any]:
    """Load a condition by its single-letter ID (A-I) or by filename stem."""
    cond_dir = CONFIG_ROOT / "conditions"
    matches = sorted(cond_dir.glob(f"{condition_id}_*.yaml"))
    if not matches:
        matches = sorted(cond_dir.glob(f"{condition_id}.yaml"))
    if not matches:
        available = ", ".join(sorted(p.stem for p in cond_dir.glob("*.yaml")))
        raise ConfigError(f"no condition {condition_id!r}. Available: {available}")
    if len(matches) > 1:
        raise ConfigError(f"ambiguous condition {condition_id!r}: {[p.name for p in matches]}")
    return load_yaml(matches[0])


def list_conditions() -> list[str]:
    return sorted(p.stem for p in (CONFIG_ROOT / "conditions").glob("*.yaml"))


# --------------------------------------------------------------------------------------
# Languages
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    script: str
    tier: str
    resource_rank: int
    unicode_blocks: tuple[str, ...] = ()
    flores_code: str | None = None
    base_language: str | None = None

    @property
    def is_code_mixed(self) -> bool:
        return self.tier == "code_mixed"

    @property
    def is_indic(self) -> bool:
        return self.tier in {"high", "medium", "low", "code_mixed"}


@dataclass(frozen=True)
class LanguageSet:
    languages: tuple[Language, ...]
    phase0_subset: tuple[str, ...] = ()

    def __iter__(self):
        return iter(self.languages)

    def __len__(self) -> int:
        return len(self.languages)

    def by_code(self, code: str) -> Language:
        for lang in self.languages:
            if lang.code == code:
                return lang
        raise ConfigError(f"unknown language code {code!r}")

    def codes(self) -> list[str]:
        return [lang.code for lang in self.languages]

    def phase0(self) -> list[Language]:
        return [self.by_code(c) for c in self.phase0_subset]

    def by_tier(self, tier: str) -> list[Language]:
        return [lang for lang in self.languages if lang.tier == tier]

    def indic(self) -> list[Language]:
        return [lang for lang in self.languages if lang.is_indic]

    def script_groups(self) -> dict[str, list[Language]]:
        """Languages grouped by script.

        Used for the script/language control (ARCHITECTURE.md §6.2): Hindi/Marathi share
        Devanagari, Bengali/Assamese share the Bengali script, so any group with more than
        one member lets us hold script constant while varying language.
        """
        groups: dict[str, list[Language]] = {}
        for lang in self.languages:
            groups.setdefault(lang.script, []).append(lang)
        return groups


def load_languages(path: str | Path = "languages.yaml") -> LanguageSet:
    data = load_yaml(path)
    langs = tuple(
        Language(
            code=entry["code"],
            name=entry["name"],
            script=entry["script"],
            tier=entry["tier"],
            resource_rank=entry["resource_rank"],
            unicode_blocks=tuple(entry.get("unicode_blocks", ())),
            flores_code=entry.get("flores_code"),
            base_language=entry.get("base_language"),
        )
        for entry in data["languages"]
    )
    return LanguageSet(languages=langs, phase0_subset=tuple(data.get("phase0_subset", ())))


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


def config_hash(config: dict[str, Any]) -> str:
    """Stable 12-char hash of a config. Artifacts are addressed by this."""
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass
class Manifest:
    """Written beside every artifact. Makes a stage skippable and a result reproducible."""

    stage: str
    config: dict[str, Any]
    outputs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "config_hash": config_hash(self.config),
            "config": self.config,
            "git_sha": git_sha(),
            "indicquant_version": __import__("indicquant").__version__,
            "outputs": self.outputs,
            "metrics": self.metrics,
            "notes": self.notes,
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path
