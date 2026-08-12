"""Unit tests for the shared config base class in blipss.models.base"""

import pytest
from pydantic import Field, ValidationError

from blipss.models.base import BlipssConfigModel


class _SampleConfig(BlipssConfigModel):
    """Minimal config model used to exercise the null-dropping behaviour of the base class."""

    required_field: int = Field(description="Field without a default")
    defaulted_int: int = Field(default=7, description="Field with a scalar default")
    defaulted_list: list[str] = Field(default_factory=lambda: [".png"], description="Field with a factory default")


@pytest.mark.parametrize(
    ("field_name", "expected"),
    [
        ("defaulted_int", 7),
        ("defaulted_list", [".png"]),
    ],
    ids=["scalar_default", "factory_default"],
)
def test_none_value_falls_back_to_declared_default(field_name: str, expected: int | list[str]) -> None:
    """A field supplied as None is dropped so the field's declared default applies."""
    cfg = _SampleConfig.model_validate({"required_field": 1, field_name: None})
    assert getattr(cfg, field_name) == expected


def test_explicit_values_are_preserved() -> None:
    """Non-None values are passed through untouched by the null-dropping validator."""
    cfg = _SampleConfig(required_field=1, defaulted_int=42, defaulted_list=[".pdf", ".png"])
    assert cfg.defaulted_int == 42
    assert cfg.defaulted_list == [".pdf", ".png"]


def test_none_for_required_field_raises_missing_field_error() -> None:
    """A required field supplied as None is dropped and therefore reported as missing."""
    with pytest.raises(ValidationError, match="Field required"):
        _SampleConfig.model_validate({"required_field": None})


def test_non_dict_input_is_forwarded_unchanged() -> None:
    """A non-dict input bypasses the null-dropping branch and is left for pydantic's own validation to reject."""
    with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
        _SampleConfig.model_validate("not-a-mapping")
