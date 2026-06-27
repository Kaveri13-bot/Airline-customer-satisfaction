def create_features(df):
    df = df.copy()

    # -------------------------
    # Clean column names
    # -------------------------
    df.columns = df.columns.str.strip()

    # -------------------------
    # Total Delay
    # -------------------------
    df["TotalDelay"] = (
        df["Departure Delay"] +
        df["Arrival Delay"]
    )

    # -------------------------
    # Average Service Rating
    # -------------------------
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

    df["AverageServiceRating"] = df[service_cols].mean(axis=1)

    # -------------------------
    # Digital Experience Score
    # -------------------------
    df["DigitalExperience"] = (
        df["Ease of Online Booking"] +
        df["Online Boarding"] +
        df["In-flight Wifi Service"]
    ) / 3

    # -------------------------
    # Airport Experience Score
    # -------------------------
    df["AirportExperience"] = (
        df["Departure and Arrival Time Convenience"] +
        df["Check-in Service"] +
        df["Gate Location"]
    ) / 3

    # -------------------------
    # Comfort Score
    # -------------------------
    df["ComfortScore"] = (
        df["Seat Comfort"] +
        df["Leg Room Service"] +
        df["Cleanliness"]
    ) / 3

    # -------------------------
    # Flight Experience Score
    # -------------------------
    df["FlightExperience"] = (
        df["In-flight Service"] +
        df["In-flight Entertainment"] +
        df["Food and Drink"]
    ) / 3

    # -------------------------
    # Frequent Traveler Flag
    # -------------------------
    df["FrequentTraveler"] = (
        df["Customer Type"]
        .map({
            "Loyal Customer": 1,
            "disloyal Customer": 0
        })
    )

    # -------------------------
    # Long Distance Flight Flag
    # -------------------------
    df["LongDistanceFlight"] = (
        df["Flight Distance"] > 2000
    ).astype(int)

    # -------------------------
    # Business Traveler Flag
    # -------------------------
    df["BusinessTraveler"] = (
        df["Type of Travel"] == "Business travel"
    ).astype(int)

    # -------------------------
    # Encode Target
    # -------------------------
    if "Satisfaction" in df.columns:
        df["Satisfaction"] = df["Satisfaction"].map({
            "Satisfied": 1,
            "Neutral or Dissatisfied": 0
        })

    return df