import joblib

model = joblib.load(
    "models/best_role_model.pkl"
)

encoder = joblib.load(
    "models/role_label_encoder.pkl"
)

samples = [

"python sql spark kafka airflow aws etl pipeline",

"power bi tableau excel sql dashboard reporting",

"machine learning tensorflow pytorch nlp deep learning",

"spring boot java microservices api mysql",

"langchain llama rag vector database prompt engineering"

]

for text in samples:

    pred = model.predict([text])[0]

    role = encoder.inverse_transform(
        [pred]
    )[0]

    print("\nSkills:")
    print(text)

    print("Prediction:")
    print(role)
