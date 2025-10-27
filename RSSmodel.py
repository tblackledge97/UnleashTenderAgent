import os
import json
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from keywords_file import keywords


load_dotenv()
OUTPUT_DIR = "TenderAusAgent_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, f"tender_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

HUBSPOT_TOKEN = os.getenv("HUBSPOT_TOKEN")
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "").strip()
NOTION_DB_ID = os.getenv("NOTION_DB_ID", "").strip()

RSS_URLS = ["https://www.tenders.gov.au/public_data/rss/rss.xml"]

# Load ML model and vectorizer
MODEL_PATH = "RSS_tender_relevance_model.pkl"
VECTORIZER_PATH = "RSS_tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; TenderBot/1.0; +https://unleashlive.com)'
}

# logging functions (from gpt)
def log(message: str):
    """Simple console + file logger."""
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now()}] {message}\n")


def safe_get(element, tag, default=""):
    """Safely extract text from an XML element."""
    try:
        node = element.find(tag)
        return node.text.strip() if node is not None and node.text else default
    except Exception:
        return default


def calculate_keyword_scores(tender):
    """Calculate weighted keyword score."""
    text = (tender.get('title', '') + " " + tender.get('description', '')).lower()
    matched = []
    total = 0

    # changes KEYWORDS to keywords to bring from other file
    for keyword, weight in keywords.items():
        if keyword in text:
            matched.append(f"{keyword} (x{weight})")
            total += weight

    return total, matched


def get_RSS():
    """
    Get the RSS feed and filter the data that come through it.
    """
    found_matches = []
    processed_log = "processed_links.txt"

    # look for processed links file, this is how we know if the tenders are new or not
    # via the tender url link (as unique, acts as primary key)
    try:
        with open(processed_log, "r") as f:
            processed_links = {line.strip() for line in f}
    except FileNotFoundError:
        processed_links = set()
        log("No previous log file found, starting fresh.")

    # go through RSS urls and get the data
    for rss_url in RSS_URLS:
        log(f"Fetching RSS feed: {rss_url}")
        try:
            resp = requests.get(rss_url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            log(f"ERROR fetching RSS feed: {e}")
            continue

        try:
            root = ET.fromstring(resp.text)
            items = root.findall(".//item")
        except ET.ParseError as e:
            log(f"ERROR parsing RSS XML: {e}")
            continue

        # from the RSS take the information available
        # from the RSS we only get a summary, a link and a title
        # start filtering the data
        for item in items:
            try:
                link = safe_get(item, "link")
                if not link or link in processed_links:
                    continue

                title = safe_get(item, "title")
                description = safe_get(item, "description")

                total_score, matched = calculate_keyword_scores({
                    "title": title,
                    "description": description
                })

                if total_score >= 10:
                    found_matches.append({
                        "title": title,
                        "description": description,
                        "url": link,
                        "total_score": total_score,
                        "matched_keywords": matched
                    })
                    processed_links.add(link)

            except Exception as e:
                log(f"ERROR processing item: {e}")
                continue

    # Update log file of processed links
    try:
        with open(processed_log, "a") as f:
            for tender in found_matches:
                f.write(tender["url"] + "\n")
    except Exception as e:
        log(f"ERROR updating processed links file: {e}")

    log(f"Found {len(found_matches)} new tenders.")
    return sorted(found_matches, key=lambda x: x["total_score"], reverse=True)


def post_to_hubspot(tender):
    if not HUBSPOT_TOKEN:
        log("Missing HubSpot token.")
        return False

    url = "https://api.hubspot.com/crm/v3/objects/deals"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {HUBSPOT_TOKEN}"}

    # for title I want to only display after the :
    # for desctiption I want to remove the <p> from the front and the </p> from the back
    payload = {
        "properties": {
            # split the title into 2 parts, before the : and after (if there is one)
            # return the second half
            "dealname": tender.get("Title", "No Title").split(":", 1)[1].strip() if ":" in tender.get("Title", "") else tender.get("Title", "No Title"),
            "dealstage": "appointmentscheduled",
            # remove the <p> and </p> if they are present
            "tender_description": tender.get("description", "")[3:-4] if tender.get("description", "").startswith("<p>") and tender.get("description", "").endswith("</p>") else tender.get("description", ""),
            "keyword_score": float(tender.get("keyword_score", 0)),

            "url": tender.get("url", "N/A"),

            "ml_recommendation": str(tender['ml_recommendation']),
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        if resp.status_code == 201:
            log(f"HubSpot: Created deal for '{tender['Title']}'")
            return True
        else:
            log(f"HubSpot error {resp.status_code}: {resp.text}")
            return False
    except requests.RequestException as e:
        log(f"ERROR posting to HubSpot: {e}")
        return False


def post_to_notion(tender):
    if not NOTION_TOKEN:
        log("Missing Notion token.")
        return False

    url = "https://api.notion.com/v1/pages"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28"
    }

    # for title I want to only display after the :
    # for desctiption I want to remove the <p> from the front and the </p> from the back
    payload = {
        "parent": {"database_id": NOTION_DB_ID},
        "properties": {
            "Title": {
                "title": [
                    {
                        "text": {
                            "content": (
                                tender.get("Title", "No Title").split(":", 1)[1].strip()
                                if ":" in tender.get("Title", "")
                                else tender.get("Title", "No Title")
                            )
                        }
                    }
                ]
            },
            "tenderdecription": {
                "rich_text": [
                    {
                        "text": {
                            "content": (
                                tender.get("description", "")
                                .removeprefix("<p>")
                                .removesuffix("</p>")[:2000]
                            )
                        }
                    }
                ]
            },
            "keyword_score": {"number": float(tender.get("keyword_score", 0))},
            "url": {"url": tender.get("url", "N/A")},

            "MLRecommendation": {
                "rich_text": [{"text": {"content": str(tender.get("ml_recommendation", ""))}}]
            },
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        if resp.status_code in (200, 201):
            log(f"Notion: Added '{tender['Title']}'")
            return True
        else:
            log(f"Notion error {resp.status_code}: {resp.text}")
            return False
    except requests.RequestException as e:
        log(f"ERROR posting to Notion: {e}")
        return False


def formatTender(tender):
    """Format the data so it is better readable for a machine learning model"""
    # Remove unwanted HTML tags etc from the description
    soup = BeautifulSoup(tender.get("description", ""), "html.parser")
    description = soup.get_text(strip=True)

    formatted_tender = {
        # title
        "Title": (
            tender.get("title", "No Title").split(":", 1)[1].strip()
            if ":" in tender.get("title", "")
            else tender.get("title", "No Title")
        ),

        # description
        "description": description[:2000],

        # keyword score
        "keyword_score": float(tender.get("total_score", 0)),
        # url
        "url": tender.get("url", "N/A"),
    }
    return formatted_tender


def predict_tender_relevance(tender, model, vectorizer):
    """
    Given a tender dict with title, description, and keyword_score,
    return True/False for whether it's relevant.
    """
    text = tender.get('Title', '') + ' ' + tender.get('description', '')
    X_text = vectorizer.transform([text])
    X_kw = np.array([[tender.get('keyword_score', 0)]])  # keyword score feature
    X_combined = hstack([X_text, csr_matrix(X_kw)])

    prediction = model.predict(X_combined)[0]
    # prob = model.predict_proba(X_combined)[0][1]  # optional probability
    return bool(prediction)


def main():
    log("=== Tender Keyword Scanner Started ===")
    tenders = get_RSS()

    if not tenders:
        log("No tenders found in RSS feed.")
        return

    for tender in tenders:
        formatted = formatTender(tender)
        is_relevant = predict_tender_relevance(formatted, model, vectorizer)
        formatted["ml_recommendation"] = f"{'True' if is_relevant else 'False'}"

        log(f"Prediction for '{tender['title']}': {is_relevant}")

        if is_relevant or tender.get('total_score', 0) > 10:
            post_to_hubspot(formatted)
            post_to_notion(formatted)

    log("=== Tender Keyword Scanner Completed ===")


if __name__ == "__main__":
    main()
