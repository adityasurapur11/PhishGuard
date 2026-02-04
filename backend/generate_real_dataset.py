import requests
import pandas as pd
import random

PHISHING_OUT = "../datasets/phishing_urls.csv"
SAFE_OUT = "../datasets/safe_urls.csv"


# ---------------- PHISHING DATA SOURCES ----------------

def download_phishtank():
    print("Downloading PhishTank data...")
    try:
        url = "https://data.phishtank.com/data/online-valid.csv"
        df = pd.read_csv(url)
        return df["url"].dropna().tolist()
    except:
        print("✖ PhishTank failed")
        return []


def download_openphish():
    print("Downloading OpenPhish data...")
    try:
        url = "https://openphish.com/feed.txt"
        data = requests.get(url, timeout=10).text.strip().split("\n")
        return data
    except:
        print("✖ OpenPhish failed")
        return []


def download_malwaredomains():
    print("Downloading MalwareDomains data...")
    try:
        url = "https://mirror1.malwaredomains.com/files/justdomains"
        data = requests.get(url, timeout=10).text.strip().split("\n")
        return data
    except:
        print("✖ MalwareDomains failed")
        return []


# ---------------- SAFE URL SOURCES (BACKUP INCLUDED) ----------------

def download_top_sites():
    print("\nDownloading SAFE websites (multiple sources)...\n")
    safe_urls = []

    # 1. Alexa Top 1M (mirror repo)
    try:
        alexa_url = "https://raw.githubusercontent.com/awesomedata/awesome-public-datasets/master/datasets/alexa-top-1m.csv"
        alexa_df = pd.read_csv(alexa_url, header=None)
        urls = ["https://" + str(d).strip() for d in alexa_df[1].tolist()]
        safe_urls += urls
        print("✔ Alexa Top Sites Loaded")
    except:
        print("✖ Alexa Top Sites FAILED")

    # 2. Majestic Million
    try:
        majestic_url = "https://downloads.majestic.com/majestic_million.csv"
        maj_df = pd.read_csv(majestic_url)
        urls = ["https://" + str(d).strip() for d in maj_df["Domain"].dropna().tolist()]
        safe_urls += urls
        print("✔ Majestic Million Loaded")
    except:
        print("✖ Majestic Million FAILED")

    # 3. DomCop Top Websites (mirror)
    try:
        dom_url = "https://raw.githubusercontent.com/mahirfaisal/domcop-top-websites/master/top10k.csv"
        dom_df = pd.read_csv(dom_url)
        urls = ["https://" + str(d).strip() for d in dom_df["domain"].dropna().tolist()]
        safe_urls += urls
        print("✔ DomCop Top Sites Loaded")
    except:
        print("✖ DomCop FAILED")

    print(f"\nTotal SAFE URLs collected so far: {len(safe_urls)}")
    return safe_urls


# ---------------- MAIN GENERATION ----------------

def generate_real_dataset():
    print("\n========== DOWNLOADING REAL DATASETS ==========\n")

    phishing = []
    safe = []

    # ---- PHISHING ----
    phishing += download_phishtank()
    phishing += download_openphish()
    phishing += download_malwaredomains()

    phishing = list(set(phishing))  # remove duplicates
    random.shuffle(phishing)
    phishing = phishing[:20000]

    print(f"\n✔ FINAL Phishing URLs: {len(phishing)}")

    # ---- SAFE ----
    safe = download_top_sites()

    safe = list(set(safe))
    random.shuffle(safe)
    safe = safe[:20000]

    print(f"✔ FINAL Safe URLs: {len(safe)}")

    # ---- SAVE ----
    print("\nSaving datasets...")
    pd.DataFrame(phishing).to_csv(PHISHING_OUT, index=False, header=False)
    pd.DataFrame(safe).to_csv(SAFE_OUT, index=False, header=False)

    print("\n========== DONE ==========")
    print(f"Phishing URLs saved: {len(phishing)}")
    print(f"Safe URLs saved: {len(safe)}")


if __name__ == "__main__":
    generate_real_dataset()
