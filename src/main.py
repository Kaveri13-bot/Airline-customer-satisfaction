from data_ingestion import split_and_save_data, load_train_data, load_test_data
from data_preprocessing import split_features_target,split_train_validation, build_preprocessor, prepare_test_features 
from model_training import train_model,build_model
from feature_engineering import create_features
from model_evaluation import evaluate_model
from model_prediction import generate_test_predictions
from config import TRAIN_DATA_PATH, TEST_DATA_PATH
import pandas as pd

def run_pipeline():

    # Step 0: Create train/test split
    print("Splitting dataset...")
    split_and_save_data()

    # Load data
    print("Loading data...")
    df_train = load_train_data()
    df_test = load_test_data()

    # Feature engineering
    print("Performing feature engineering...")

    def create_features(df):

        df = df.copy()

        # Clean column names
        df.columns = df.columns.str.strip()

        # Total Delay
        df["TotalDelay"] = (
            df["Departure Delay"] +
            df["Arrival Delay"]
        )

        # Service-related columns
        service_cols = [
            "Check-in Service",
            "Online Boarding",
            "On-board Service",
            "Seat Comfort",
            "Leg Room Service",
            "Cleanliness",
            "Food and Drink",
            "In-flight Service",
            "In-flight Wifi Service",
            "In-flight Entertainment",
            "Baggage Handling"
        ]

        # Average Service Rating
        df["AverageServiceRating"] = df[service_cols].mean(axis=1)

        # Digital Experience Score
        df["DigitalExperience"] = (
            df["Ease of Online Booking"] +
            df["Online Boarding"] +
            df["In-flight Wifi Service"]
        ) / 3

        # Airport Experience Score
        df["AirportExperience"] = (
            df["Departure and Arrival Time Convenience"] +
            df["Check-in Service"] +
            df["Gate Location"]
        ) / 3

        # Comfort Score
        df["ComfortScore"] = (
            df["Seat Comfort"] +
            df["Leg Room Service"] +
            df["Cleanliness"]
        ) / 3

        # Flight Experience Score
        df["FlightExperience"] = (
            df["In-flight Service"] +
            df["In-flight Entertainment"] +
            df["Food and Drink"]
        ) / 3

        # Frequent Traveler
        df["FrequentTraveler"] = df["Customer Type"].map({
            "Loyal Customer": 1,
            "disloyal Customer": 0
        })

        # Long Distance Flight
        df["LongDistanceFlight"] = (
            pd.to_numeric(
                df["Flight Distance"],
                errors="coerce"
            ) > 2000
        ).astype(int)

        # Business Traveler
        df["BusinessTraveler"] = (
            df["Type of Travel"] == "Business travel"
        ).astype(int)

        # Encode Target
        if "Satisfaction" in df.columns:
            df["Satisfaction"] = df["Satisfaction"].map({
                "Satisfied": 1,
                "Neutral or Dissatisfied": 0
            })

        return df

    # Apply feature engineering
    df_train = create_features(df_train)
    df_test = create_features(df_test)

    # Split features and target
    print("Preparing features and target...")
    X, y = split_features_target(df_train)

    # Build preprocessor
    print("Building preprocessor...")
    preprocessor = build_preprocessor(X)

    # Split into training and validation
    print("Splitting train and validation...")
    X_train, X_val, y_train, y_val = split_train_validation(X, y)

    # Build and train model
    print("Training model...")
    model = build_model(preprocessor)
    trained_model = train_model(model, X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    evaluate_model(trained_model, X_val, y_val)

    # Test predictions
    print("Generating predictions...")
    X_test = prepare_test_features(df_test)
    generate_test_predictions(trained_model, X_test)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()