import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

torch.cuda.is_available(), torch.cuda.get_device_name(0)

device = torch.device("cuda:0")
torch.cuda.set_device(device)

torch.cuda.current_device()





# ---------- CONFIG ----------
CSV_PATH = "test_spam_data.csv"   # change this
TEXT_COLUMN = "text"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# ----------------------------

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def classify_text(text: str) -> str:
    prompt = f"""You are a spam classifier.
Classify the SMS strictly as SPAM or HAM
Reply with only one word.

Message: "{text}"
Label:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=0.0
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(decoded)
    # print(decoded.split("Label:")[-1])
    print('54',decoded.split("Label:")[-1].strip().split()[0])
    
    
    label = decoded.split("Label:")[-1].strip().split()[0].upper()

    return label if label in ["SPAM", "HAM"] else "HAM"
    # return label if label in ["SPAM", "HAM"] else "HAM"


# Read CSV
# Read original CSV
df = pd.read_csv(CSV_PATH)

records = []

for idx, row in df.iterrows():
    text = row[TEXT_COLUMN]
    gt = row["label"] #.upper()

    pred = classify_text(text)
    records.append({
        "label": gt,
        "text": text,
        "prediction": pred
    })

# Create new DataFrame
pred_df = pd.DataFrame(records)

# Save new CSV
OUTPUT_CSV = "spam_ham_predictions.csv"
pred_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved predictions to {OUTPUT_CSV}")


from sklearn.metrics import accuracy_score, confusion_matrix
# Read prediction CSV
df_pred = pd.read_csv(OUTPUT_CSV)

# Normalize labels
df_pred["label"] = df_pred["label"].str.upper()
df_pred["prediction"] = df_pred["prediction"].str.upper()

# Filter valid predictions only
valid_df = df_pred[df_pred["prediction"].isin(["HAM", "SPAM"])]

y_true = valid_df["label"]
y_pred = valid_df["prediction"]

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=["HAM", "SPAM"])

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix (GT rows, Pred columns)")
print("        HAM   SPAM")
print(f"HAM   {cm[0][0]:5d}  {cm[0][1]:5d}")
print(f"SPAM  {cm[1][0]:5d}  {cm[1][1]:5d}")




