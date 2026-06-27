import pandas as pd
from sklearn.model_selection import train_test_split
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, TEST_SIZE, RANDOM_STATE


def split_and_save_data():
    """Load dataset and split into train/test"""
    df = pd.read_csv(r"C:\data\Airline_info.csv")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print("Train and Test files created successfully!")

    
def load_train_data():
    return pd.read_csv(TRAIN_DATA_PATH)



def load_test_data():
    return pd.read_csv(TEST_DATA_PATH)

if __name__ == "__main__":
    split_and_save_data()