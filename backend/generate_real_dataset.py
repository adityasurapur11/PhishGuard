import requests
import pandas as pd
import random
import io
import os
import zipfile
import string

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "..", "datasets")
PHISHING_OUT = os.path.join(DATASETS_DIR, "phishing_urls.csv")
SAFE_OUT = os.path.join(DATASETS_DIR, "safe_urls.csv")
SUSPICIOUS_OUT = os.path.join(DATASETS_DIR, "suspicious_urls.csv")

HEADERS = {"User-Agent": "Mozilla/5.0"}

def generate_suspicious_mutation(url):
    """
    Creates a 'Suspicious' version of a safe URL by:
    1. Removing a random symbol/char (user request)
    2. Swapping characters (typo)
    """
    try:
        # parsed = url.split("://")[1]
        if len(url) < 5: return url
        
        # Method 1: Remove a character (User specific request)
        idx = random.randint(0, len(url) - 1)
        return url[:idx] + url[idx+1:]
    except:
        return url + "v"

def generate_phishing_rules_url():
    """Generates a URL that strictly meets the Phishing conditions provided"""
    brands = ['google', 'paypal', 'amazon', 'apple', 'microsoft', 'netflix', 'facebook']
    tlds = ['.tk', '.ml', '.cf', '.ga']
    ip_bases = ['192.168', '10.0', '172.16']
    
    case = random.randint(1, 6)
    
    if case == 1: # Misspelled
        brand = random.choice(brands)
        spoof = brand.replace('o', '0').replace('l', 'I').replace('a', '@')
        return f"http://{spoof}.com/login"
        
    elif case == 2: # Extra words + hyphens
        brand = random.choice(brands)
        words = ['verify', 'account', 'secure', 'update', 'alert']
        return f"https://{brand}-{random.choice(words)}-login.com"
        
    elif case == 3: # Subdomain trick
        brand = random.choice(brands)
        return f"https://{brand}.com.verify-user.info"
        
    elif case == 4: # IP Address
        return f"http://{random.choice(ip_bases)}.{random.randint(0,255)}.{random.randint(0,255)}/admin"
        
    elif case == 5: # Suspicious TLD
        brand = random.choice(brands)
        return f"http://{brand}-secure{random.choice(tlds)}"
        
    elif case == 6: # File extension
        return f"http://files-upload.com/invoice.exe"

def generate_massive_dataset():
    print("\n========== 🚀 GENERATING MASSIVE DATASET (100k+ Safe) ==========\n")
    if not os.path.exists(DATASETS_DIR): os.makedirs(DATASETS_DIR)

    # 1. GET 100,000+ SAFE URLs
    print("📥 Downloading Top 1 Million Safe Sites...")
    safe_urls = []
    
    # Try Tranco List (Best source)
    try:
        resp = requests.get("https://tranco-list.eu/top-1m.csv.zip", headers=HEADERS, timeout=60)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            with z.open(z.namelist()[0]) as f:
                # Read top 150,000 to be safe
                df = pd.read_csv(f, header=None, nrows=150000)
                safe_urls = ["https://" + str(d).strip() for d in df[1].tolist()]
    except Exception as e:
        print(f"⚠️ Tranco download failed ({e}). using fallback...")
        # Fallback: Generate safe urls from a base list to ensure we hit 100k
        base_domains = ['google', 'youtube', 'facebook', 'baidu', 'wikipedia', 'qq', 'taobao', 'yahoo', 'tmall', 'amazon']
        tlds = ['.com', '.org', '.net', '.io', '.co', '.in', '.de', '.jp']
        for i in range(12000): # Generate enough
            for b in base_domains:
                safe_urls.append(f"https://{b}{i}{random.choice(tlds)}")

    # Ensure we have at least 100k
    safe_urls = list(set(safe_urls))
    if len(safe_urls) > 100000:
        safe_urls = safe_urls[:100000] # Cap at 100k exact if preferred, or keep all
    
    print(f"   ✔ Loaded {len(safe_urls)} Safe URLs")

    # 2. GENERATE PHISHING (Rule Based)
    print("💀 Generating Phishing URLs (Rule Matches)...")
    # We want a good amount, say 20k, to train the model to recognize these patterns
    phishing_urls = [generate_phishing_rules_url() for _ in range(20000)]
    
    # Add some real ones too
    try:
        url = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt"
        resp = requests.get(url, timeout=20)
        real_phish = [line.strip() for line in resp.text.split("\n") if len(line) > 10 and "." in line]
        phishing_urls += real_phish[:5000]
    except: pass
    
    print(f"   ✔ Loaded {len(phishing_urls)} Phishing URLs")

    # 3. GENERATE SUSPICIOUS (Mutations of Safe)
    print("🧪 Generating Suspicious URLs (Mutations)...")
    # Take 20k safe URLs and mutate them (remove chars, etc)
    suspicious_urls = []
    subset_safe = safe_urls[:20000]
    for url in subset_safe:
        suspicious_urls.append(generate_suspicious_mutation(url))
        
    print(f"   ✔ Created {len(suspicious_urls)} Suspicious URLs")

    # Save
    print("💾 Saving to CSV...")
    pd.DataFrame(safe_urls).to_csv(SAFE_OUT, index=False, header=False)
    pd.DataFrame(phishing_urls).to_csv(PHISHING_OUT, index=False, header=False)
    pd.DataFrame(suspicious_urls).to_csv(SUSPICIOUS_OUT, index=False, header=False)
    
    print("\n✅ DATASET COMPLETE.")

if __name__ == "__main__":
    generate_massive_dataset()
