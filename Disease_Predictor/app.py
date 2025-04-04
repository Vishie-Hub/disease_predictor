from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

data_path = os.path.dirname(__file__)
df_symptoms = pd.read_csv(os.path.join(data_path, "symptom_dataset.csv"))
df_descriptions = pd.read_csv(os.path.join(data_path, "symptom_description.csv"))
df_precautions = pd.read_csv(os.path.join(data_path, "symptom_precaution.csv"))

disease_symptom_map = {}
for _, row in df_symptoms.iterrows():
    disease = row["Disease"]
    symptoms = row.drop("Disease").dropna().str.strip().str.lower().tolist()
    if disease not in disease_symptom_map:
        disease_symptom_map[disease] = []
    disease_symptom_map[disease].extend(symptoms)
disease_symptom_map = {k: list(set(v)) for k, v in disease_symptom_map.items()}

description_lookup = dict(zip(df_descriptions["Disease"], df_descriptions["Description"]))
precaution_lookup = {
    row["Disease"]: [row[f"Precaution_{i}"] for i in range(1, 5) if pd.notna(row[f"Precaution_{i}"])]
    for _, row in df_precautions.iterrows()
}

def predict_diseases(user_symptoms):
    user_symptoms = [s.strip().lower() for s in user_symptoms]
    scores = {}
    for disease, symptoms in disease_symptom_map.items():
        match_count = sum(symptom in user_symptoms for symptom in symptoms)
        if match_count > 0:
            scores[disease] = match_count
    top_diseases = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    results = []
    for disease, score in top_diseases:
        desc = description_lookup.get(disease, "No description available.")
        precautions = precaution_lookup.get(disease, [])
        results.append({
            "Disease": disease,
            "Matching Symptoms": score,
            "Description": desc,
            "Precautions": precautions
        })
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        symptom_input = request.form["symptoms"]
        symptoms = [s.strip() for s in symptom_input.split(",")]
        results = predict_diseases(symptoms)
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
