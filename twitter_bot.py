import numpy as np
import pandas as pd
import time
import re
import joblib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score


# FEATURE ENGINEERING

def engineer_features(df, tfidf, fit_tfidf=False):
    d = df.copy()

    d["followers_count"]  = pd.to_numeric(d.get("followers_count",  0), errors="coerce").fillna(0).astype(int)
    d["friends_count"]    = pd.to_numeric(d.get("friends_count",    0), errors="coerce").fillna(0).astype(int)
    d["listed_count"]     = pd.to_numeric(d.get("listed_count",     0), errors="coerce").fillna(0).astype(int)
    d["statuses_count"]   = pd.to_numeric(d.get("statuses_count",   0), errors="coerce").fillna(0).astype(int)

    fav_col = "favourites_count" if "favourites_count" in d.columns else "favorites_count"
    d["favourites_count"] = pd.to_numeric(d.get(fav_col, 0), errors="coerce").fillna(0).astype(int)

    d["verified"] = d["verified"].apply(
        lambda x: 1 if str(x).strip().upper() in ["TRUE", "1"] else 0)

    bag = (r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget'
           r'|expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA'
           r'|nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone'
           r'|genie|emoji|joke|troll|droop|every|wow|cheese|yeah|bio|magic|wizard|face')

    d["kw_name"]        = d["name"].astype(str).str.contains(bag, case=False, na=False).astype(int)
    d["kw_description"] = d["description"].astype(str).str.contains(bag, case=False, na=False).astype(int)
    d["kw_screen_name"] = d["screen_name"].astype(str).str.contains(bag, case=False, na=False).astype(int)
    d["kw_status"]      = d["status"].astype(str).str.contains(bag, case=False, na=False).astype(int)
    d["is_buzzfeed"]    = d["description"].astype(str).str.contains("buzzfeed", case=False, na=False).astype(int)
    d["listed_gt_16k"]  = (d["listed_count"] > 16000).astype(int)

    d["follower_ratio"]   = d["followers_count"] / (d["friends_count"] + 1)
    d["bio_length"]       = d["description"].fillna("").apply(len)
    d["bio_is_empty"]     = (d["description"].fillna("") == "").astype(int)
    d["username_digits"]  = d["screen_name"].apply(lambda x: sum(c.isdigit() for c in str(x)))
    d["username_len"]     = d["screen_name"].apply(lambda x: len(str(x)))
    d["username_has_bot"] = d["screen_name"].str.contains("bot", case=False, na=False).astype(int)
    d["default_profile_image"] = d["default_profile_image"].apply(
        lambda x: 1 if str(x).strip().upper() in ["TRUE", "1"] else 0)

    bio_text = d["description"].fillna("")
    if fit_tfidf:
        tfidf.fit(bio_text)
    tfidf_matrix = tfidf.transform(bio_text).toarray()
    tfidf_cols   = [f"bio_tfidf_{w}" for w in tfidf.get_feature_names_out()]
    tfidf_df     = pd.DataFrame(tfidf_matrix, columns=tfidf_cols, index=d.index)
    d = pd.concat([d, tfidf_df], axis=1)

    feature_cols = [
        "followers_count", "friends_count", "statuses_count",
        "favourites_count", "listed_count",
        "follower_ratio", "bio_length", "bio_is_empty",
        "username_digits", "username_len", "username_has_bot",
        "verified", "default_profile_image",
        "kw_name", "kw_description", "kw_screen_name", "kw_status",
        "is_buzzfeed", "listed_gt_16k",
    ] + tfidf_cols

    return d[feature_cols].fillna(0)



# MAIN

if __name__ == "__main__":
    start = time.time()

    # Load data
    train_df = pd.read_csv("training_data_2_csv_UTF.csv", encoding="latin-1", on_bad_lines="skip")
    train_df = train_df.rename(columns={"ï»¿id": "id"})
    train_df["bot"] = pd.to_numeric(train_df["bot"], errors="coerce")
    train_df = train_df.dropna(subset=["bot"])

    print(f"Labeled samples: {len(train_df)}")
    print(train_df["bot"].value_counts())

    # Train/val split
    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df["bot"]
    )

    #  tfidf and model
    tfidf = TfidfVectorizer(max_features=50, stop_words="english")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        class_weight="balanced", random_state=42, n_jobs=-1
    )

    # Train
    X_train = engineer_features(train_split, tfidf, fit_tfidf=True)
    y_train = train_split["bot"].astype(int)
    model.fit(X_train, y_train)
    print(f"✅ Trained on {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Validate
    X_val  = engineer_features(val_split, tfidf, fit_tfidf=False)
    y_val  = val_split["bot"].astype(int)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    print(f"Validation Accuracy: {metrics.accuracy_score(y_val, y_pred):.2%}")
    print(f"ROC-AUC:             {metrics.roc_auc_score(y_val, y_prob):.2%}")
    print(metrics.classification_report(y_val, y_pred, target_names=["Human", "Bot"]))

    # Retrain on full data
    X_all = engineer_features(train_df, tfidf, fit_tfidf=True)
    y_all = train_df["bot"].astype(int)
    model.fit(X_all, y_all)

   
    joblib.dump(model, "rf_model.pkl")
    joblib.dump(tfidf, "tfidf.pkl")
    print(" Saved: rf_model.pkl and tfidf.pkl")

