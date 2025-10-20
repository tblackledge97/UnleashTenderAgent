import json
import pickle
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from dotenv import load_dotenv
import requests
import xml.etree.ElementTree as ET
from keywords_file import keywords

load_dotenv()
HUBSPOT_TOKEN = os.getenv("HUBSPOT_TOKEN")
TENDERINFO_API_URL = "https://tenderdetailapi.tendersinfo.net/api/BOQ/GetMyTenders"
ORIGIN_HEADER = "unleashlive.com"
SUBNO_PARAM = 329595

NOTION_TOKEN = os.getenv("NOTION_TOKEN", "").strip()
NOTION_DB_ID = os.getenv("NOTION_DB_ID").strip()

OUTPUT_DIR = "TenderAusAgent_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_keyword_scores(tender):
    """
    Calculates a score based on the predefined, weighted keyword dictionary.
    """
    score = 0
    # combine the text and dictionary
    text = (tender.get('title', '') + " " + tender.get('description', '')).lower()

    for keyword, weight in keywords.items():
        if keyword in text:
            score += weight

    return score


def predict_relevance(new_tenders, model, desc_vectorizer, other_vectorizer):
    """
    Uses the trained model to predict relevance.
    """
    # Vectoriser has already been fitted so to our vocab
    # so use .transform() not .fit_transform()
    new_descriptions = [tender.get('description', '') for tender in new_tenders]
    new_categories_agencies = [tender.get('category', '') + " " + tender.get('agency', '') for tender in new_tenders]

    new_keyword_scores = [calculate_keyword_scores(tender) for tender in new_tenders]
    new_X_keyword = csr_matrix(np.array(new_keyword_scores)).T

    new_X_desc = desc_vectorizer.transform(new_descriptions)
    new_X_other = other_vectorizer.transform(new_categories_agencies)

    # combine the data
    new_X_combined = hstack([new_X_desc, new_X_other, new_X_keyword])

    predictions = model.predict(new_X_combined)

    return predictions


def fetch_new_tenders(from_index=0, to_index=100, tender_type="New"):
    """fetches new tenders from TenderInfo API."""

    # define the POST request
    # Headers
    headers = {
        "Origin": ORIGIN_HEADER,
        "Content-Type": "application/json"
    }

    # request params
    params = {
        "subno": SUBNO_PARAM
    }

    # body parameters
    body = {
        "From": from_index,
        "To": to_index,
        "Type": tender_type
    }

    print(f"Fetching tenders from index {from_index} to {to_index}")

    try:
        # send the POST request
        response = requests.post(
            TENDERINFO_API_URL,
            headers=headers,
            params=params,
            json=body
        )
        response.raise_for_status()

        data = response.json()

        print(json.dumps(data, indent=2))

        # extract the tenders
        if data.get("isSuccess") and "TENDERS" in data.get("Data", {}):
            print(f"Successful: {len(data['Data']['TENDERS'])} tenders fetched.")
            return data["Data"]["TENDERS"]

        print("API reported successfully but no Tenders data found.")
        return []

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return []


def format_tender_data(raw_tender):
    """Helper function to translate TenderInfo API field names
    into what we are expecting.
    fields:
    - tendersbrief    :   description
    - companyname    :   agency
    - industryname   :   category (keyword)
    - originalsource :   URL
    
    """

    category_parts = [
        raw_tender.get('productname', ''),
        raw_tender.get('keywordname', ''),
        raw_tender.get('industryname', '')
    ]

    category_text = " ".join([p for p in category_parts if p.strip()])

    formatted_tender = {
        # title
        "title": raw_tender.get('tendertype', '') or raw_tender.get('companyname', 'N/A'),

        # description
        "description": raw_tender.get('tendersbrief', 'N/A'),

        # agency
        "agency": raw_tender.get('companyname', 'N/A'),

        # category
        "category": category_text,

        # URL
        "url": raw_tender.get('originalsource', 'N/A'),

        # contract value
        "value": raw_tender.get('tendervalue', 'N/A'),

        # countryname
        "countryname": raw_tender.get('countryname', 'N/A')

    }

    return formatted_tender


def load_trained_model(model_path='model.pkl', desc_path='desc_vectorizer.pkl', other_path='other_vectorizer.pkl'):
    """Loads the saved ML model and vectorizers."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(desc_path, 'rb') as f:
            desc_vectorizer = pickle.load(f)
        with open(other_path, 'rb') as f:
            other_vectorizer = pickle.load(f)
        return model, desc_vectorizer, other_vectorizer
    except FileNotFoundError:
        print("Model files don't exist")
        return None, None, None


def post_to_hubspot(match_data):
    """
    Posts a single matched tender to HS
    """
    if not HUBSPOT_TOKEN:
        print("Error, no token.")
        return False

    url = "https://api.hubspot.com/crm/v3/objects/deals"


    # define headers and content-type with auth code
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HUBSPOT_TOKEN}"
    }

    # map the data to HS
    # add URL when the URL is available from the RSS data
    payload = {
        "properties": {
            # Deal name
            "dealname": match_data['title'],

            # Deal stage
            # we dont really have this but it is mandatory for HS
            "dealstage": "appointmentscheduled",

            # ML reccomendation
            "ml_recommendation": str(match_data['ml_recommendation']),

            # Use get in case it doesnt exist
            # Description
            "tender_description": match_data.get('description', 'N/A'),

            # our keyword scoring
            "keyword_score": str(match_data['keyword_score']),

            # Use get in case it doesnt exist
            # agency
            "agency": match_data.get('agency', 'N/A'),

            # URL so we can visit the site of the tender
            # is URL the correct working for hubspot????
            "url": match_data.get('url', 'N/A'),

            # country name
            "country_name": match_data.get('countryname', 'N/A')

            # how much is the tender 
            # hubspot_totalcontractvalue
            #"hs_tcv": match_data.get('value')
        }
    }

    # send to HS (POST)
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 201:
        print(f"sucess: {match_data['title']} created in Hubspot")
        return True
    else:
        print(f"failure: {response.status_code}")
        print(f"Error: {response.text}")
        return False


def post_to_notion(match_data):
    """Posts a single matched tender to notion API"""
    if not NOTION_TOKEN:
        print("No Nothion token")
        return False

    url = "https://api.notion.com/v1/pages"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
    }

    payload = {
        "parent": {"database_id": NOTION_DB_ID},
        "properties": {
            # check what notion calls these things
            # dealname
            "Title": {  
                "title": [{"text": {"content": match_data.get("title", "No Title")}}]
            },

            # deal stage
            # this was needed for HS, is it for NOtion, if not leave blank

            # this will need to be a custom field
            # ML recommendation
            "MLRecommendation": {
                "rich_text": [{"text": {"content": str(match_data.get("ml_recommendation", ""))}}]
            },

            # description
            "tenderdecription": {
                "rich_text": [{"text": {"content": match_data.get("description", "N/A")}}]
            },

            # keyword scoring
            # this will need to be a custom field
            "keyword_score": {
                "number": float(match_data.get("keyword_score", 0))
            },

            # agency
            "agency": {
                "rich_text": [{"text": {"content": match_data.get("agency", "N/A")}}]
            },

            # URL
            "url": {
                "url": match_data.get("url", None)
            },
            # country name
            "countryname": {
               "rich_text": [{"text": {"content": match_data.get('countryname', 'N/A')}}]
            }

            # how much is the tender 
            # hubspot_totalcontractvalue
            #"value": match_data.get('value', 'N/A')

        }
    }
    # send to notion
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == [200, 201]:
        print(f"success: {match_data['title']} created in HS.")
    else:
        print(f"failure {response.status_code}")
        print(f"{response.text}")
        return False


# def log_to_local(tenders):
#     """Saves Tenders locally
#     This is not functionality that I think is required but
#     just a good visual for demo"""
#     df = pd.DataFrame(tenders)
#     csv_path = os.path.join(OUTPUT_DIR)
#     html_path = os.path.join(OUTPUT_DIR)

#     df.to_csv(csv_path, index=False)
#     df.to_html(html_path, index=False)


if __name__ == '__main__':
    # load the models and vectorizers
    model, desc_vectorizer, other_vectorizer = load_trained_model()

    if model:
        # new tenders, this is where the scraper will feed data to
        # NEW_TENDERS = [
        #     {"title": "IT Support RFP", "description": "Seeking a vendor to provide IT support and network maintenance for a government office.", "category": "IT Support", "agency": "Department of Finance"},
        #     {"title": "Office Furniture Supply", "description": "Request for proposals for the supply and installation of new office furniture.", "category": "Office Furniture", "agency": "Department of Health"},
        #     {"title": "Data Analytics Solution", "description": "The Department of Defence requires a new software solution for data analysis and reporting.", "category": "Software Development", "agency": "Department of Defence"},
        #     {"title": "Landscaping Services", "description": "Call for tenders for landscaping and gardening services at a public park.", "category": "Landscaping", "agency": "Department of the Environment"}
        # 
        # fetch raw data
        raw_tenders = fetch_new_tenders(from_index=0, to_index=100, tender_type="New")

        # format the data
        NEW_TENDERS = [format_tender_data(raw) for raw in raw_tenders]

        if not NEW_TENDERS:
            print("No new tenders")
        else:
            # now predict
            predictions = predict_relevance(NEW_TENDERS, model, desc_vectorizer, other_vectorizer)

            print("-- Live Agent Predictions Report ---")
            final_matches = []

            for tender_data, prediction in zip(NEW_TENDERS, predictions):

                keyword_score = calculate_keyword_scores(tender_data)

                is_final_match = prediction or (keyword_score >= 10)

                relevance_status = "RELEVANT" if prediction else "Not relevant"

                # Print the prediction for the console report
                print(f"Title: {tender_data['title']} | AI: {relevance_status} | Score: {keyword_score} | FINAL MATCH: {is_final_match}")

                if is_final_match:
                    # Log the success...
                    # Structure the output data with new fields and logic:
                    match_details = {
                        "title": tender_data.get('title'),
                        "description": tender_data.get('description'),
                        "url": tender_data.get('url', 'N/A'),
                        "agency": tender_data.get('agency'),
                        "keyword_score": keyword_score,
                        "ml_recommendation": bool(prediction),
                        "status": "New Lead",
                        "companyname": "companyname",
                        #"value": tender_data.get('value')
                    }
                    final_matches.append(match_details)

            # post to HS
            if final_matches:
                print("---Posting to HS---")
                for match in final_matches:
                    post_to_hubspot(match)
                    # post to notion to, perhaphs look at al script
                    post_to_notion(match)
                #log_to_local(final_matches)

            else:
                print("No leads found worth posting")
