"""
metrics/core/metric_config.py

Declarative configuration schema for metrics.

This is METADATA only — describes what parameters a metric accepts.
Used for:
- CLI argument generation
- UI forms
- Documentation
- Validation (optional)

NOT used for storing actual runtime values (those go in metric._params).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParamSpec:
    """
    Specification for a single parameter.

    Attributes:
        type: The Python type (int, float, str, bool, tuple, etc.)
        default: Default value (None means required unless optional=True)
        optional: If True, parameter can be omitted even without default
        choices: Valid values (for enum-like parameters)
        help: Human-readable description
        min_value: Minimum value (for numeric types)
        max_value: Maximum value (for numeric types)
    """

    type: type
    default: Any = None
    optional: bool = False
    choices: list[Any] | None = None
    help: str | None = None
    min_value: int | float | None = None
    max_value: int | float | None = None

    @property
    def is_required(self) -> bool:
        """Parameter is required if no default and not optional."""
        return self.default is None and not self.optional


class MetricConfig:
    """
    Declarative configuration schema for a metric.

    Example usage:
    ```python
    config = MetricConfig({
        "n_steps": {
            "type": int,
            "default": 21,
            "help": "Number of interpolation steps",
            "min_value": 2,
        },
        "sigma": {
            "type": float,
            "default": 0.05,
            "help": "Noise standard deviation",
            "min_value": 0.0,
        },
        "target_class": {
            "type": int,
            "default": None,
            "optional": True,
            "help": "Target class index (None = use predicted)",
        },
        "kind": {
            "type": str,
            "default": "regression",
            "choices": ["regression", "classification"],
        },
    })
    ```
    """

    def __init__(self, params: dict[str, dict[str, Any]]) -> None:
        """
        Initialize from a dictionary of parameter specifications.

        Args:
            params: Dict mapping parameter names to their specs.
                Each spec is a dict with keys: type, default, optional, choices, help, etc.
        """
        self._raw = params
        self._specs: dict[str, ParamSpec] = {}

        for name, spec_dict in params.items():
            self._specs[name] = ParamSpec(
                type=spec_dict.get("type", str),
                default=spec_dict.get("default"),
                optional=spec_dict.get("optional", False),
                choices=spec_dict.get("choices"),
                help=spec_dict.get("help"),
                min_value=spec_dict.get("min_value"),
                max_value=spec_dict.get("max_value"),
            )

    @property
    def params(self) -> dict[str, ParamSpec]:
        """Get all parameter specifications."""
        return self._specs.copy()

    @property
    def raw(self) -> dict[str, dict[str, Any]]:
        """Get raw parameter dict (for serialization)."""
        return self._raw.copy()

    def get(self, name: str) -> ParamSpec | None:
        """Get spec for a specific parameter."""
        return self._specs.get(name)

    def required_params(self) -> list[str]:
        """List parameter names that are required."""
        return [name for name, spec in self._specs.items() if spec.is_required]

    def optional_params(self) -> list[str]:
        """List parameter names that are optional."""
        return [name for name, spec in self._specs.items() if not spec.is_required]

    def validate(self, values: dict[str, Any]) -> list[str]:
        """
        Validate parameter values against the schema.

        Args:
            values: Dict of parameter name -> value

        Returns:
            List of error messages (empty if valid)
        """
        errors: list[str] = []

        # Check required params
        for name in self.required_params():
            if name not in values:
                errors.append(f"Missing required parameter: {name}")

        # Check types and constraints
        for name, value in values.items():
            spec = self._specs.get(name)
            if spec is None:
                # Unknown parameter - could warn or ignore
                continue

            if value is None and spec.optional:
                continue

            # Type check (basic)
            if not isinstance(value, spec.type) and value is not None:
                # Allow int for float
                if not (spec.type is float and isinstance(value, int)):
                    errors.append(
                        f"Parameter {name}: expected {spec.type.__name__}, "
                        f"got {type(value).__name__}"
                    )

            # Choices check
            if spec.choices is not None and value not in spec.choices:
                errors.append(f"Parameter {name}: must be one of {spec.choices}, got {value}")

            # Range checks
            if spec.min_value is not None and value is not None:
                if value < spec.min_value:
                    errors.append(f"Parameter {name}: must be >= {spec.min_value}, got {value}")
            if spec.max_value is not None and value is not None:
                if value > spec.max_value:
                    errors.append(f"Parameter {name}: must be <= {spec.max_value}, got {value}")

        return errors

    def with_defaults(self, values: dict[str, Any]) -> dict[str, Any]:
        """
        Return values dict with defaults filled in for missing params.

        Args:
            values: Provided parameter values

        Returns:
            New dict with defaults applied
        """
        result = {}
        for name, spec in self._specs.items():
            if name in values:
                result[name] = values[name]
            elif spec.default is not None:
                result[name] = spec.default
            elif spec.optional:
                result[name] = None
        return result

    def __repr__(self) -> str:
        param_strs = [
            f"{name}: {getattr(spec.type, '__name__', repr(spec.type))}"
            for name, spec in self._specs.items()
        ]
        return f"MetricConfig({', '.join(param_strs)})"
