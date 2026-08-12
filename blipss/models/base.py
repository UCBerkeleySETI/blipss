"""Shared base class for blipss config models."""

from typing import Any

from pydantic import BaseModel, model_validator


class BlipssConfigModel(BaseModel):
    """Base for config models where a blank/null YAML field is treated as absent, falling back to the field's declared default."""

    @model_validator(mode="before")
    @classmethod
    def _drop_none_values(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data
