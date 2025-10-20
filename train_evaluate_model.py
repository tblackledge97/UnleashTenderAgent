# Training and testing + evaluating the tenders model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix
import json
import numpy as np
import pickle
import joblib

FILE_PATH = "/Users/tristanblackledge/TenderAusAgent/ai-agent-project/tender_scraper/tenders_data.json"

keywords = {
    "IT": 4,
    "ai": 5,
    "artificial intelligence": 5,
    "machine learning": 4,
    "ml": 4,
    "deep learning": 3,
    "predictive maintenance": 5,
    "computer vision": 5,
    "object detection": 4,
    "edge computing": 4,
    "real-time analytics": 5,
    "realtime analytics": 5,
    "digital twin": 4,
    "automation": 4,
    "computer vision platform": 5,
    "video analytics": 5,
    "live video": 5,
    "streaming": 5,
    "stream": 5,
    "real-time": 5,
    "realtime": 5,
    "video": 4,
    "image": 4,
    "photo": 3,
    "cctv": 4,
    "camera": 3,
    "gis": 4,
    "geographic information systems": 4,
    "remote sensing": 4,
    "lidar": 5,
    "orthomosaic": 3,
    "spatial analytics": 4,
    "hazard mapping": 3,
    "evacuation modelling": 2,
    "early warning systems": 3,
    "biodiversity monitoring": 2,
    "infrastructure": 5,
    "roads": 4,
    "streets": 3,
    "corridors": 5,
    "smart cities": 4,
    "utilities": 5,
    "powerline": 5,
    "stormwater": 3,
    "parking": 3,
    "asset management": 5,
    "asset lifecycle": 4,
    "condition assessment": 5,
    "inspection": 5,
    "resilience": 3,
    "retrofit": 2,
    "decarbonisation": 2,
    "climate adaptation": 3,
    "renewable energy": 5,
    "renewables": 5,
    "solar": 5,
    "wind": 5,
    "wind turbine": 5,
    "turbine": 4,
    "operational efficiency": 4,
    "manufacturing": 3,
    "supply chain visibility": 3,
    "flow monitoring": 3,
    "counting": 3,
    "disaster": 4,
    "fire": 4,
    "flood": 4,
    "emergency response": 4,
    "emergency": 4,
    "public safety": 4,
    "vessel monitoring": 3,
    "remote operations": 4,
    "drone": 5,
    "drones": 5,
    "uav": 5,
    "unmanned aerial systems": 5,
    "uav inspection": 5,
    "drone video analytics": 5,
    "uav program manager": 4,
    "remote site inspection": 5,
    "sovereign industrial priorities": 3,
    "skilling stream": 3,
    "exports stream": 3,
    "security stream": 3,
    # my added words for testing
    "software development": 4,
    "web application": 4,
    "IT services": 4,
    "cyber security": 5,
    "cloud computing": 4,
    "data analysis": 5,
    "software": 4,
    "data": 3
}


# load the data
def load_tenders_data(file_path):
    """Load the tender data, some is real data, some is fake."""
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


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


def prepare_data(tenders):
    """Combines features and splits data into Train/Validation/Test sets."""

    # Feature Extraction
    descriptions = [tender.get('description', '') for tender in tenders]
    categories_agencies = [tender.get('category', '') + " " + tender.get('agency', '') for tender in tenders]
    keyword_scores = np.array([calculate_keyword_scores(tender) for tender in tenders]).reshape(-1, 1)

    # label extraction
    labels = np.array([tender.get('is_relevant') for tender in tenders])

    # Combine All Text into a Single Array
    full_text = [tenders[i].get('description', '') + ' ' + 
                 tenders[i].get('category', '') + ' ' + 
                 tenders[i].get('agency', '') for i in range(len(tenders))]

    # Split indices first to keep all features aligned
    indices = np.arange(len(tenders))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, labels, random_state=42, test_size=0.4, stratify=labels)

    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, random_state=42, test_size=0.5, stratify=y_temp)

    # Fit Vectorizer on TRAINING TEXT ONLY
    full_vectorizer = TfidfVectorizer(ngram_range=(1,2))

    # Use the combined text strings corresponding to the training indices
    train_text = [full_text[i] for i in train_idx]
    full_vectorizer.fit(train_text)

    # Transform ALL splits using the fitted vectorizer
    X_train = full_vectorizer.transform(train_text)
    X_val = full_vectorizer.transform([full_text[i] for i in val_idx])
    X_test = full_vectorizer.transform([full_text[i] for i in test_idx])

    # Add Keyword Scores
    X_train_kw = keyword_scores[train_idx]
    X_val_kw = keyword_scores[val_idx]
    X_test_kw = keyword_scores[test_idx]

    # Final Combined Feature Matrices (hstack)
    X_train = hstack([X_train, csr_matrix(X_train_kw)])
    X_val = hstack([X_val, csr_matrix(X_val_kw)])
    X_test = hstack([X_test, csr_matrix(X_test_kw)])

    return X_train, X_val, X_test, y_train, y_val, y_test, full_vectorizer


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Trains the logistic regression model
    Evaluates against the test data."""

    model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, C=0.5)
    model.fit(X_train, y_train)

    # predict from test data
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', pos_label=True)
    recall = recall_score(y_test, y_pred, average='binary', pos_label=True)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print("CM:\n", confusion_matrix(y_test, y_pred))

    return model


if __name__ == "__main__":
    tenders_data = load_tenders_data(FILE_PATH)

    # prep and split data
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = prepare_data(tenders_data)

    # train model
    final_model = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Save the trained model and vectorizer
    joblib.dump(final_model, 'tender_relevance_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
