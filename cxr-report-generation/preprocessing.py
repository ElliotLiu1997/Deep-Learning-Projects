# read data
import pandas as pd

report = pd.read_csv('indiana_reports.csv')
figure = pd.read_csv('indiana_projections.csv')

# split labels
def split_labels(x):
    if pd.isna(x):
        return []
    return x.split(";")

report["Problems_list"] = report["Problems"].apply(split_labels)

import re

def clean_label(label):
    label = label.lower().strip()
    label = re.sub(r"[^\w\s]", " ", label)
    label = re.sub(r"\s+", " ", label)
    return label

report["Problems_list"] = report["Problems_list"].apply(
    lambda labels: [clean_label(l) for l in labels if l != ""]
)

all_labels = []
for labels in report["Problems_list"]:
    all_labels.extend(labels)

from collections import Counter

counter = Counter(all_labels)

print(len(report))
print(len(figure))

print(counter)

# filter labels
selected_labels = [
    "normal",
    "cardiomegaly",
    "atelectasis",
    "pleural effusion",
    "lung opacity",
    "nodule",
    "airspace disease",
    "atherosclerosis",
    "granulomatous disease",
    "calcified granuloma",
    "calcinosis",
    "cicatrix",
    "scoliosis"
]

def map_label(label):
    if "atelectasis" in label:
        return "atelectasis"
    if "opacity" in label or "density" in label:
        return "lung opacity"
    return label

def process_labels(labels):
    new_labels = []
    
    for l in labels:
        l = map_label(l)
        
        if l in selected_labels:
            new_labels.append(l)
    
    return list(set(new_labels))

report["Filtered_labels"] = report["Problems_list"].apply(process_labels)
report["label_len"] = report["Filtered_labels"].apply(len)
report = report[report["label_len"] > 0]

# re-check
counter = Counter()

for labels in report["Filtered_labels"]:
    counter.update(labels)

print(counter)

import re
import pandas as pd

def safe_text(x):
    return "" if pd.isna(x) else str(x)

def normalize_text(text: str) -> str:
    text = safe_text(text).lower()
    repl = {
        "c/w": "consistent_with",
        "r/l": "right_left",
        "and/or": "and_or",
        "w/o": "without",
        "w/": "with",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text

def clean_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"x{3,}", " ", text)
    text = re.sub(r"\b\d+\.\b", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\-/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_tokens(tokens):
    out = []
    for t in tokens:
        t = t.strip().rstrip(".")
        if not t:
            continue
        if re.match(r"x-\w+", t):
            continue
        if re.fullmatch(r"x{3,}", t):
            continue
        out.append(t)
    return out

def preprocess_report(findings, impression, mode="combo"):
    f = safe_text(findings)
    i = safe_text(impression)

    if mode == "findings":
        raw = f
    elif mode == "impression":
        raw = i
    else:
        raw = (f + ". " + i).strip(". ").strip()

    text = clean_text(raw)
    tokens = clean_tokens(text.split())

    if len(tokens) < 3:
        return {"clean_text": "", "tokens": [], "sentences": []}

    clean = " ".join(tokens)
    sentences = [s.strip() for s in re.split(r"\.+", clean) if s.strip()]
    return {"clean_text": clean, "tokens": tokens, "sentences": sentences}

def process_dataframe(df, mode="combo"):
    df = df.copy()
    results = df.apply(
        lambda r: preprocess_report(r.get("findings"), r.get("impression"), mode=mode),
        axis=1,
    )
    df["clean_text"] = results.apply(lambda x: x["clean_text"])
    df["tokens"] = results.apply(lambda x: x["tokens"])
    df["sentences"] = results.apply(lambda x: x["sentences"])
    return df

report = process_dataframe(report, mode="combo")

figure = figure[figure["projection"] == "Frontal"]
data = report.merge(figure, on="uid")
print(data[["uid", "filename", "clean_text"]].head())
print(len(data))

data["is_normal"] = data["Filtered_labels"].apply(
    lambda x: 1 if x == ["normal"] else 0
)
from sklearn.model_selection import train_test_split

# 70% train, 30% temp
train_df, temp_df = train_test_split(
    data,
    test_size=0.3,
    random_state=123,
    stratify=data["is_normal"]
)

# temp → val + test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=123,
    stratify=temp_df["is_normal"]
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

data = pd.concat([train_df, val_df, test_df])
print(data["split"].value_counts())



# classification vec
data["label_vec"] = data["Filtered_labels"].apply(
    lambda labels: [1 if l in labels else 0 for l in selected_labels]
)

# caption
data = data[data["tokens"].apply(lambda x: isinstance(x, list) and len(x) >= 3)].copy()
print(data["split"].value_counts())

# vocab
from collections import Counter
counter = Counter()
for tokens in data.loc[data["split"] == "train", "tokens"]:
    counter.update(tokens)

vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + [w for w, c in counter.items() if c >= 2]
word2idx = {w: i for i, w in enumerate(vocab)}

def encode(tokens, max_len=90):
    seq = [word2idx["<SOS>"]]
    seq += [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]
    seq.append(word2idx["<EOS>"])
    if len(seq) < max_len:
        seq += [word2idx["<PAD>"]] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
        seq[-1] = word2idx["<EOS>"]
    return seq

data["caption_seq"] = data["tokens"].apply(encode)

data.to_csv("info.csv", index=False)
