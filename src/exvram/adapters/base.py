from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from typing import Protocol

from ..contracts import AdapterAvailability, AdapterInfo, ExperimentConfig


class AdapterProtocol(Protocol):
    """Stable boundary implemented by every external runtime adapter."""

    info: AdapterInfo

    def availability(self) -> AdapterAvailability:
        ...

    def plan(self, config: ExperimentConfig) -> dict[str, object]:
        ...


@dataclass(frozen=True)
class ExternalAdapter:
    info: AdapterInfo
    import_names: tuple[str, ...] = ()
    executable_names: tuple[str, ...] = ()
    installation_note: str = "Install the external project using its official instructions."

    def availability(self) -> AdapterAvailability:
        for module in self.import_names:
            if importlib.util.find_spec(module) is not None:
                return AdapterAvailability(
                    self.info.name, "available", f"Python module {module} found", module
                )
        for executable in self.executable_names:
            if shutil.which(executable) is not None:
                return AdapterAvailability(
                    self.info.name, "available", f"Executable {executable} found", executable
                )
        return AdapterAvailability(
            self.info.name,
            "unavailable",
            "External runtime is not installed; ExVRAM does not vendor it.",
        )

    def plan(self, config: ExperimentConfig) -> dict[str, object]:
        return {
            "adapter": self.info.name,
            "status": "integration_only",
            "availability": self.availability().to_dict(),
            "installation_note": self.installation_note,
            "config_id": config.id,
            "copied_code": self.info.copied_code,
        }
