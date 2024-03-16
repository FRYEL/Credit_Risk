import pandas as pd

df = pd.read_csv("data/cleaned_data.csv", low_memory = False)

non_zero_annual_inc = df[df['annual_inc'] > 0]['annual_inc']

# Calculate the minimum value for annual income
min_annual_inc = non_zero_annual_inc.min()

# Print the minimum value
print("Minimum Annual Income (excluding zeros): ${:,.2f}".format(min_annual_inc))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
 pass