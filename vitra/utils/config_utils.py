import json

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
    """Load configuration file with support for parent configs."""
    with open(config_file) as f:
        _config = json.load(f)
    
    config = {}
    if _config.get("parent"):
        deep_update(config, load_config(_config["parent"]))
    deep_update(config, _config)
    return config