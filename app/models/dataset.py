from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, FilePath


def _validate_json_file(value: Path) -> Path:
    if value.suffix.lower() != ".json":
        raise ValueError(f"Dataset file must be a JSON file: {value}")
    return value


JsonFile = Annotated[FilePath, AfterValidator(_validate_json_file)]


class DatasetFileInput(BaseModel):
    dataset_file: JsonFile
