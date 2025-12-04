import json
import os
from huggingface_hub import hf_hub_download

def deep_update(d1, d2):
    """Deep update d1 with d2, recursively merging nested dictionaries."""
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            assert isinstance(d1[k], dict), f"Cannot merge dict with non-dict for key {k}"
            deep_update(d1[k], d2[k])
        else:
            d1[k] = d2[k]
    return d1


def load_config(config_file):
    """Load configuration file with support for parent configs and Hugging Face Hub."""
    # Check if config_file is a Hugging Face repo (format: "username/repo-name:filename")
    from_huggingface = False
    if ":" in config_file and "/" in config_file.split(":")[0] and not os.path.exists(config_file):
        # Parse Hugging Face repo format: "username/repo-name:config.json"
        repo_id, filename = config_file.split(":", 1)
        print(f"Loading config from Hugging Face Hub: {repo_id}/{filename}")
        config_path = hf_hub_download(repo_id=repo_id, filename=f"{filename}")
        from_huggingface = True
    elif "/" in config_file and not os.path.exists(config_file) and not config_file.endswith(".json"):
        # If format is "username/repo-name", default to "config.json"
        print(f"Loading config from Hugging Face Hub: {config_file}/configs/config.json")
        config_path = hf_hub_download(repo_id=config_file, filename="configs/config.json")
        from_huggingface = True
    else:
        # Local file path
        config_path = config_file
        from_huggingface = False

    with open(config_path) as f:
        _config = json.load(f)

    if from_huggingface:
        _config["model_load_path"] = config_file
        _config['statistics_path'] = config_file

    config = {}
    if _config.get("parent"):
        deep_update(config, load_config(_config["parent"]))
    deep_update(config, _config)
    return config