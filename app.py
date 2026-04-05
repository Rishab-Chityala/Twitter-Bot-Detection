import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================================================
# LOAD MODEL — uses rf_model.pkl and tfidf.pkl (no class needed)
# ================================================================
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf

model, tfidf = load_model()

# ================================================================
# FEATURE ENGINEERING
# ================================================================
def engineer_features(df):
    d = df.copy()

    
    for col in ["followers_count", "friends_count", "listed_count", "statuses_count"]:
        if col not in d.columns:
            d[col] = 0
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(int)

    fav_col = "favourites_count" if "favourites_count" in d.columns else "favorites_count"
    if fav_col not in d.columns:
        d["favourites_count"] = 0
    else:
        d["favourites_count"] = pd.to_numeric(d[fav_col], errors="coerce").fillna(0).astype(int)

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

    bio_text     = d["description"].fillna("")
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

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(page_title="Twitter Bot Detector", page_icon="🤖", layout="wide")
st.title("🤖 Twitter Bot Detector")
st.markdown("Upload a CSV of Twitter accounts to detect how many are bots.")

with st.sidebar:
    st.header("How to use")
    st.markdown("""
    1. Upload a CSV file
    2. Click **Analyze for Bots**
    3. Download results
    """)
    st.divider()
    st.markdown("**Model performance**")
    st.metric("Validation Accuracy", "91.4%")
    st.metric("ROC-AUC", "96.7%")

# ================================================================
# UPLOAD + PREDICT
# ================================================================
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin-1", on_bad_lines="skip")
    df = df.rename(columns={"favorites_count": "favourites_count"})

    st.subheader("Preview")
    st.dataframe(df.head(5), use_container_width=True)

    if st.button("🔍 Analyze for Bots", type="primary"):
        with st.spinner("Running bot detection..."):
            try:
                X     = engineer_features(df)
                preds = model.predict(X)
                probs = model.predict_proba(X)[:, 1]

                df["predicted_bot"]   = preds
                df["bot_probability"] = (probs * 100).round(1)
                df["verdict"]         = df["predicted_bot"].map({1: "🤖 Bot", 0: "✅ Human"})

                total       = len(df)
                bot_count   = int(df["predicted_bot"].sum())
                human_count = total - bot_count
                bot_pct     = round(bot_count / total * 100, 1)

                st.divider()
                st.subheader("📊 Results")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total",        total)
                c2.metric("🤖 Bots",      bot_count)
                c3.metric("✅ Humans",     human_count)
                c4.metric("Bot %",        f"{bot_pct}%")

                st.subheader("🎯 Bot probability distribution")
                st.bar_chart(df["bot_probability"].value_counts().sort_index())

                st.subheader("📋 Account breakdown")
                filter_opt = st.radio("Show:", ["All", "Bots only", "Humans only"], horizontal=True)
                filtered = df.copy()
                if filter_opt == "Bots only":
                    filtered = df[df["predicted_bot"] == 1]
                elif filter_opt == "Humans only":
                    filtered = df[df["predicted_bot"] == 0]

                show_cols = [c for c in ["screen_name", "followers_count",
                             "friends_count", "statuses_count",
                             "bot_probability", "verdict"] if c in df.columns]
                st.dataframe(
                    filtered[show_cols].sort_values("bot_probability", ascending=False
                    ).reset_index(drop=True),
                    use_container_width=True
                )

                st.download_button(
                    "⬇️ Download results",
                    filtered[show_cols].to_csv(index=False),
                    "bot_results.csv", "text/csv"
                )

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("👆 Upload a CSV file to get started")