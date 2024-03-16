import pandas as pd
import sys

# Check for python version greater 2.8
if sys.version_info[0:2] < (2, 8):
    raise Exception("Requires python 2.8 or higher")

df = pd.read_csv("data/cleaned_data.csv", low_memory=False)

if __name__ == '__main__':
    pass
