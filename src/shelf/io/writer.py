"""CSV writer."""

from pathlib import Path

import pandas as pd

from shelf.schema import OUTPUT_COLUMNS, PriceTag


def write_csv(tags: list[PriceTag], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    rows = [t.to_dict() for t in tags]
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(output_path, index=False)
    return output_path
