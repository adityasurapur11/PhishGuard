import re
import numpy as np
from urllib.parse import urlparse

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    features = {
        "length": len(url),
        "num_digits": sum(c.isdigit() for c in url),
        "num_hyphens": url.count("-"),
        "num_slashes": url.count("/"),
        "num_dots": url.count("."),
        "has_https": 1 if url.startswith("https") else 0,
        "domain_length": len(domain),
        "num_subdomains": domain.count(".") - 1 if "." in domain else 0,
        "path_length": len(path),
        "suspicious_words": (
            "login" in url or "verify" in url or "update" in url or
            "free" in url or "secure" in url or "bank" in url or
            "offer" in url
        ),
        "contains_ip": bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain)),
    }

    # Convert booleans to int
    for k in features:
        if isinstance(features[k], bool):
            features[k] = int(features[k])

    return np.array(list(features.values()), dtype=np.float32)
