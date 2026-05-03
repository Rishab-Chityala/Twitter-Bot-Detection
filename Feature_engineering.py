import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from dateutil import parser as dateparser

# LOADING DATA
train = pd.read_csv("training_data_2_csv_UTF.csv", encoding="latin-1", on_bad_lines="skip")
train = train.rename(columns={"ï»¿id": "id", "bot": "label"})

test = pd.read_csv("test_data_4_students.csv", encoding="latin-1", sep="\t", on_bad_lines="skip")
test = test.rename(columns={"bot": "label"})



test = test.rename(columns={
    "favorites_count": "favourites_count"  
})


# FEATURE 1 — Follower / Following ratio

def follower_ratio(row):
    following = row["friends_count"]
    if following == 0:
        return row["followers_count"]  
    return row["followers_count"] / following

train["follower_ratio"] = train.apply(follower_ratio, axis=1)
test["follower_ratio"]  = test.apply(follower_ratio, axis=1)


# FEATURE 2 — Post frequency (tweets per day since account created)

def account_age_days(created_at):
    try:
        created = dateparser.parse(str(created_at), ignoretz=True)
        today   = pd.Timestamp.now()
        delta   = (today - created).days
        return max(delta, 1)  
    except:
        return np.nan

def tweet_frequency(row):
    age = account_age_days(row["created_at"])
    if pd.isna(age):
        return np.nan
    return row["statuses_count"] / age

train["account_age_days"] = train["created_at"].apply(account_age_days)
train["tweet_frequency"]  = train.apply(tweet_frequency, axis=1)

test["account_age_days"]  = test["created_at"].apply(account_age_days)
test["tweet_frequency"]   = test.apply(tweet_frequency, axis=1)


# FEATURE 3 — Username patterns

def username_features(username):
    username = str(username)
    return {
        "username_len":        len(username),
        "username_digits":     sum(c.isdigit() for c in username),
        "username_underscores":username.count("_"),
        "username_has_bot":    int("bot" in username.lower()),
    }

train_uname = train["screen_name"].apply(username_features).apply(pd.Series)
test_uname  = test["screen_name"].apply(username_features).apply(pd.Series)

train = pd.concat([train, train_uname], axis=1)
test  = pd.concat([test,  test_uname],  axis=1)

# FEATURE 4 — Bio features

train["bio"]          = train["description"].fillna("")
test["bio"]           = test["description"].fillna("")

train["bio_length"]   = train["bio"].apply(len)
test["bio_length"]    = test["bio"].apply(len)

train["bio_is_empty"] = (train["bio"] == "").astype(int)
test["bio_is_empty"]  = (test["bio"]  == "").astype(int)


tfidf = TfidfVectorizer(max_features=50, stop_words="english")
tfidf.fit(train["bio"])  

train_tfidf = pd.DataFrame(
    tfidf.transform(train["bio"]).toarray(),
    columns=[f"bio_tfidf_{w}" for w in tfidf.get_feature_names_out()]
)
test_tfidf = pd.DataFrame(
    tfidf.transform(test["bio"]).toarray(),
    columns=[f"bio_tfidf_{w}" for w in tfidf.get_feature_names_out()]
)

train = pd.concat([train.reset_index(drop=True), train_tfidf], axis=1)
test  = pd.concat([test.reset_index(drop=True),  test_tfidf],  axis=1)

# FEATURE 5 — Profile flags

def to_bool_int(val):
    if isinstance(val, bool): return int(val)
    if str(val).strip().upper() in ["TRUE", "1"]:  return 1
    return 0

train["default_profile_image"] = train["default_profile_image"].apply(to_bool_int)
test["default_profile_image"]  = test["default_profile_image"].apply(to_bool_int)

train["verified"] = train["verified"].apply(to_bool_int)
test["verified"]  = test["verified"].apply(to_bool_int)


# FINAL — Select only model features

FEATURES = [
    "followers_count", "friends_count", "statuses_count",
    "favourites_count", "listed_count",
    "follower_ratio", "tweet_frequency", "account_age_days",
    "username_len", "username_digits", "username_underscores", "username_has_bot",
    "bio_length", "bio_is_empty",
    "verified", "default_profile_image",
] + [f"bio_tfidf_{w}" for w in tfidf.get_feature_names_out()]

X_train = train[FEATURES].fillna(0)
y_train = train["label"].fillna(0).astype(int)

X_test  = test[FEATURES].fillna(0)
y_test  = test["label"].fillna(0).astype(int)

print("X_train shape:", X_train.shape)
print("X_test shape: ", X_test.shape)
print("Features:", FEATURES[:10], "... +", len(FEATURES)-10, "more")
print("\nSample features:")
print(X_train.head(3).to_string())