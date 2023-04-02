from datasets import load_dataset, Dataset


def get_maestro() -> Dataset:
    dataset = load_dataset("roszcz/maestro-v1-sustain")
    return dataset
