# UnleashTenderAgent
Python scripts to receive information from an RSS feed and an API full of possible tender opportunities. These opportunities are then filtered through keyword searching and a small logistical regression model before being sent to a CRM and sales hub.

# Files
- Keywords
Keywords_file.py: The dictionary of keywords. Keys are the keywords and values are the weight associated with them
- RSS
RSSmodel.py: Run this file daily to recieve the latest opportunities from the AusTender RSS feed and filter for oppotunities.
RSS_train_model.py: Run this file only when new training data is added or parameters (such as keyword weightings) are changed. This is where the model is trained.
RSS_tenders_data.json: Training data for the RSS model.
- API
predictor.py: Run this file daily to revieve "New" tender opportunites from the TenderInfo API and filter for opportunities,
Reasoning.py: Functions that are used elsewhere to train relevance as well as generate keyword scorings for tenders that come in from the API.
train_evaluate_model.py: Where the model is trained and evaluated. Run this when training data or parameters are added or changed.
tenders_data.json: Training data for the API model.

The folder also contains various pre-trained models and vectorizers so that the RSSmodel.py and predictor.py scripts can run without training. These files should be overwritten automatically when a new model is trained and saved from the train_evaluate_model.py or RSS_train_model.py files.

Processed_links.txt is a file containing already seen URLs from the RSS feed. This prevent duplicates.





