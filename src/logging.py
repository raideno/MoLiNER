import json
import tqdm
import yaml
import logging

# from https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def save_metric(path, metrics, format="yaml"):
    """
    Save evaluation metrics to a file in YAML or JSON format.

    Args:
        path (str): Path to save the file (without extension).
        metrics (dict): Metrics dictionary.
        format (str): 'yaml' (default) or 'json'.
    """
    if format == "yaml":
        strings = yaml.dump(metrics, indent=4, sort_keys=False)
        full_path = path if path.endswith(".yaml") else path + ".yaml"
        with open(full_path, "w") as f:
            f.write(strings)
    elif format == "json":
        full_path = path if path.endswith(".json") else path + ".json"
        with open(full_path, "w") as f:
            json.dump(metrics, f, indent=4)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"[metrics saved to]: {full_path}")