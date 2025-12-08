import pandas as pd
import re

def get_stacktrace_from_logs(csv_path):
    df = pd.read_csv(csv_path)

    stacktrace_patterns = [
        r"Traceback \(most recent call last\)",           # Python
        r"^\s+File \".*\", line \d+",                     # Python stack frame
        r"^\s+at\s+[\w\.]+\([^)]*\)",                     # Java/C#/Go stack frame
        r"^\s*Exception in thread \"",                   # Java exception header
        r"^\s*\w+Exception:",                             # Generic Exception: ...
        r"^\s*\w+Error:",                                 # Generic Error: ...
    ]

    combined_pattern = "(" + "|".join(stacktrace_patterns) + ")"

    mask = df["message"].str.contains(
        combined_pattern,
        flags=re.IGNORECASE | re.MULTILINE,
        regex=True
    )

    stacktrace_rows = df[mask]
    pd.set_option('display.max_colwidth', None)
    #print(stacktrace_rows["message"])
    return stacktrace_rows["message"]

def main():
    csv_path = "dataset/RE3-OB/cartservice_f1/1/logs.csv"
    get_stacktrace_from_logs(csv_path)

if __name__ == "__main__":
    main()
