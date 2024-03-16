import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

data = pd.read_csv("data/cleaned_data.csv", low_memory = False)

print(data.head(5))


# Histogram
plt.figure(figsize=(10, 6))
plt.hist(data['loan_amnt'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Loan Amounts')
plt.show()

# Bar Chart
plt.figure(figsize=(10, 6))
data['term'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Loan Term')
plt.ylabel('Count')
plt.title('Distribution of Loan Terms')
plt.show()

# Pie Chart
plt.figure(figsize=(8, 8))

grade_counts = data['grade'].value_counts()
grade_order = sorted(grade_counts.index)
grade_counts.loc[grade_order].plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'))

plt.title('Distribution of Loan Grades')
plt.ylabel('')
plt.show()

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['grade'], y=data['int_rate'], palette='viridis')
plt.xlabel('Loan Grade')
plt.ylabel('Interest Rate')
plt.title('Interest Rate Distribution by Loan Grade')
plt.show()

# Scatterplot

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['loan_amnt'], y=data['annual_inc'], alpha=0.5)
plt.xlabel('Loan Amount')
plt.ylabel('Annual Income')
plt.title('Loan Amount vs. Annual Income')
plt.yscale('log')

median_annual_inc = data['annual_inc'].median()
plt.axhline(y=median_annual_inc, color='r', linestyle='--', label='Median Annual Income')

formatter = ScalarFormatter()
formatter.set_scientific(False)
plt.gca().yaxis.set_major_formatter(formatter)

plt.legend()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='inferno', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Stacked Bar Chart
plt.figure(figsize=(10, 6))
stacked_df_indv = data.groupby(['term', 'grade']).size().unstack()
stacked_df_indv.plot(kind='bar', stacked=True)
plt.xlabel('Loan Term')
plt.ylabel('Count of Loans')
plt.title('Distribution of Loan Grades by Term')
plt.legend(title='Loan Grade')
plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
 pass