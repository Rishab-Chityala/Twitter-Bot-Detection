import pandas as pd

# Training data
train = pd.read_csv(
    "training_data_2_csv_UTF.csv",
    encoding="latin-1",
    on_bad_lines="skip"
)
train = train.rename(columns={"ï»¿id": "id", "bot": "label"})

# Test data 
test = pd.read_csv(
    "test_data_4_students.csv",
    encoding="latin-1",
    sep="\t",
    on_bad_lines="skip"
)
test = test.rename(columns={"bot": "label"})

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Label counts:\n", train["label"].value_counts())