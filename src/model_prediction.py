import pandas as pd

def generate_test_predictions(model, X_test):

    predictions = model.predict(X_test)

    output = pd.DataFrame({
        "Prediction": predictions
    })

    output.to_csv("test_predictions.csv", index=False)

    print("Test predictions saved successfully!")