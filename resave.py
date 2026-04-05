# save this as resave.py in your project folder and run it once
import joblib

# Load old model
old = joblib.load("twitter_bot_model.pkl")

# Save components separately
joblib.dump(old.model, "rf_model.pkl")
joblib.dump(old.tfidf, "tfidf.pkl")
print("Done! rf_model.pkl and tfidf.pkl created")