
import csv
import os

FEATURES = [
    "lat",
    "lon",
    "elevation",
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "rain_14d",
    "rain_21d",
    "rain_30d",
    "humidity",
    "temperature",
    "soil_wetness",
    "month_sin",
    "month_cos"
]

TARGET = "flooded"
METADATA = ["event_id", "cluster_id", "split"]


def load_data(path="data/processed/flood_data.csv"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    X = []
    y = []
    metadata = []

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV file is empty.")

        required = FEATURES + [TARGET] + METADATA
        missing = [column for column in required
                   if column not in reader.fieldnames]

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        for line_number, row in enumerate(reader, start=2):
            try:
                features = []

                for column in FEATURES:
                    value = row[column]

                    # Preserve missing values for preprocessing.
                    if value is None or value.strip() == "":
                        if column == "soil_wetness":
                            features.append(None)
                        else:
                            raise ValueError(f"Missing {column}")
                    else:
                        features.append(float(value))

                label = int(row[TARGET])

                if label not in (0, 1):
                    raise ValueError("Target must be 0 or 1.")

                event_id = row["event_id"].strip()
                cluster_id = row["cluster_id"].strip()
                split = row["split"].strip()

                if not event_id or not cluster_id:
                    raise ValueError("Missing event or cluster ID.")

                if split not in ("train", "val", "test"):
                    raise ValueError(f"Invalid split: {split}")

                X.append(features)
                y.append(label)
                metadata.append({
                    "event_id": event_id,
                    "cluster_id": cluster_id,
                    "split": split
                })

            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(
                    f"Invalid data at CSV line {line_number}: {error}"
                ) from error

    if not X:
        raise ValueError("Dataset contains no samples.")

    return X, y, metadata
