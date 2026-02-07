import re
import numpy as np
from urllib.parse import urlparse

# Specific patterns from user requirements
SUSPICIOUS_TLDS = {'.tk', '.ml', '.cf', '.ga', '.gq', '.zip', '.top', '.xyz', '.site'}
BRAND_NAMES = {'google', 'amazon', 'paypal', 'apple', 'microsoft', 'bank', 'netflix', 'facebook', 'instagram'}
URGENCY_WORDS = {'verify', 'urgent', 'suspend', 'update', 'confirm', 'security', 'login', 'account', 'free', 'offer'}
SHORTENERS = {'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'rebrand.ly', 'is.gd', 'ow.ly'}

def extract_features(url):
    url = url.lower()
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    
    # 1. Basic Lengths
    url_len = len(url)
    domain_len = len(domain)
    path_len = len(path)
    
    # 2. Character Counts
    num_hyphens = url.count('-')
    num_dots = url.count('.')
    num_digits = sum(c.isdigit() for c in url)
    num_special = len(re.findall(r'[@?&=%]', url))
    
    # 3. Protocol
    is_https = 1 if url.startswith("https") else 0
    
    # 4. IP Address Check
    contains_ip = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain) else 0
    
    # 5. URL Shortener Check
    is_shortened = 1 if any(s in domain for s in SHORTENERS) else 0
    
    # 6. Suspicious TLD Check
    has_suspicious_tld = 1 if any(url.endswith(tld) or (tld + "/") in url for tld in SUSPICIOUS_TLDS) else 0
    
    # 7. Brand Spoofing (e.g., amazon-security.com)
    # Check if a brand name is in the domain but not the main part of it
    brand_spoofing = 0
    for brand in BRAND_NAMES:
        if brand in domain:
            # If it's not exactly brand.com or www.brand.com, it's suspicious
            if not (domain == f"{brand}.com" or domain == f"www.{brand}.com" or 
                    domain.endswith(f".{brand}.com") or domain == f"{brand}.in"):
                brand_spoofing = 1
                break
                
    # 8. Urgency Keywords
    keyword_count = sum(1 for word in URGENCY_WORDS if word in url)
    
    # 9. Subdomain Trick (many dots in domain)
    subdomain_count = domain.count('.')
    subdomain_trick = 1 if subdomain_count > 3 else 0

    # 10. File Extensions in URL
    has_file_ext = 1 if any(url.endswith(ext) for ext in ['.exe', '.zip', '.html', '.php', '.js']) else 0

    features = [
        url_len,            # 0
        domain_len,         # 1
        path_len,           # 2
        num_hyphens,        # 3
        num_dots,           # 4
        num_digits,         # 5
        num_special,        # 6
        is_https,           # 7
        contains_ip,        # 8
        is_shortened,       # 9
        has_suspicious_tld, # 10
        brand_spoofing,     # 11
        keyword_count,      # 12
        subdomain_trick,    # 13
        has_file_ext        # 14
    ]

    return np.array(features, dtype=np.float32)