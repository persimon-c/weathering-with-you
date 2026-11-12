def split_data(X, y, metadata):
    if not (len(X) == len(y) == len(metadata)):
        raise ValueError("Input lengths must match.")

    if not X:
        raise ValueError("Dataset cannot be empty.")

    splits = {
        "train": {"X": [], "y": [], "metadata": []},
        "val": {"X": [], "y": [], "metadata": []},
        "test": {"X": [], "y": [], "metadata": []}
    }

    for features, label, info in zip(X, y, metadata):
        split = info["split"]

        if split not in splits:
            raise ValueError(f"Invalid split: {split}")

        splits[split]["X"].append(features)
        splits[split]["y"].append(label)
        splits[split]["metadata"].append(info)

    return splits
