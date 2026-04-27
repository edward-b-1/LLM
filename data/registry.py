import os
from data.prepare import DATASET_DIR


def _paths(name):
    return (
        os.path.join(DATASET_DIR, f"{name}_train.bin"),
        os.path.join(DATASET_DIR, f"{name}_val.bin"),
    )


DATASETS = {
    "shakespeare":          _paths("shakespeare"),
    "shakespeare-complete": _paths("shakespeare_complete"),
    "wikipedia-en":         _paths("wikipedia"),
    "wikipedia-fr":         _paths("wikipedia_fr"),
}

# Val-only paths for eval.py
VAL_DATASETS = {name: paths[1] for name, paths in DATASETS.items()}
