import pandas as pd
import os

def get_error_code_from_trace(csv_path):
    df = pd.read_csv(csv_path, dtype=str)

    # Normalize numeric columns
    df["statusCode"] = pd.to_numeric(df["statusCode"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    # Count each unique status code, sorted by descending count
    status_code_counts = (
        df["statusCode"]
        .dropna()
        .astype(int)
        .value_counts()          # already sorted descending by default
        .to_dict()
    )

    # Keep only non-zero status codes
    error_df = df[df["statusCode"].notna() & (df["statusCode"] != 0)]

    # First occurrence of each statusCode
    first_errors = error_df.drop_duplicates(subset="statusCode", keep="first")

    error_rows = (
        first_errors[
            ["serviceName", "methodName", "operationName", "duration", "statusCode"]
        ]
        .fillna("")
        .to_dict(orient="records")
    )

    return error_rows, status_code_counts

def main():

    DATASET = "dataset/RE3-OB"

    for folder in os.listdir(DATASET):
        folder_path = os.path.join(DATASET, folder)
        if not os.path.isdir(folder_path):
            continue

        for case in os.listdir(folder_path):
            case_path = os.path.join(folder_path, case)
            print(f'Case: {case}')

            traces_path = os.path.join(case_path, "traces.csv")
            if not os.path.exists(traces_path):
                continue

            error_rows, status_code_counts = get_error_code_from_trace(traces_path)
            print(error_rows)
            print(status_code_counts)
            print()

if __name__ == "__main__":
    main()
