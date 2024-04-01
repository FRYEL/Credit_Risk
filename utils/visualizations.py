import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

colors = ['#0476df', '#50b1ff', '#0458a5', '#88cbff', '#00457a', '#032a4d', '#9e9e9e', '#828282', '#0078d6']
sns.set_palette(sns.color_palette(colors))

# Load df and unprocessed df
df = pd.read_csv("../data/cleaned_data.csv", low_memory=False)
df_legacy = pd.read_csv("../data/loan/loan.csv", low_memory=False)

# Plot interest rate distribution by loan status
plt.figure(figsize=(10, 7))
sns.boxplot(data=df, x='loan_status', y='int_rate')
plt.xlabel('Loan Status')
plt.ylabel('Interest Rate (%)')
plt.title('Interest Rate Distribution by Loan Status')
plt.xticks(rotation=45)
plt.savefig('Interest_Rate_Distribution_by_Loan_Status.png')
plt.show()

# Plot debt-to-income ratio distribution by loan status
plt.figure(figsize=(10, 7))
sns.violinplot(data=df, x='loan_status', y='dti')
plt.xlabel('Loan Status')
plt.ylabel('Debt-to-Income Ratio')
plt.title('Debt-to-Income Ratio Distribution by Loan Status')
plt.xticks(rotation=45)
plt.savefig('Debt_to_Income_Ratio_Distribution_by_Loan_Status.png')
plt.show()

# Plot verification status distribution by loan status
plt.figure(figsize=(10, 7))
sns.countplot(data=df_legacy, x='loan_status', hue='verification_status')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.title('Verification Status Distribution by Loan Status')
plt.xticks(rotation=45)
plt.legend(title='Verification Status', bbox_to_anchor=(1, 1))
plt.savefig('Verification_Status_Distribution_by_Loan_Status.png')
plt.show()

# Plot interest rate distribution
plt.figure(figsize=(10, 7))
sns.histplot(df['int_rate'], kde=True, palette=colors[3])
plt.xlabel('Interest Rate (%)')
plt.ylabel('Frequency')
plt.title('Interest Rate Distribution')
plt.savefig('Interest_Rate_Distribution.png')
plt.show()

# Plot annual income distribution with logarithmic x-axis and smaller range
plt.figure(figsize=(10, 7))
sns.histplot(data=df, x='annual_inc', kde=True, palette=colors[2])
plt.xlabel('Annual Income (Log Scale) 9,999-200,000')
plt.ylabel('Frequency')
plt.title('Annual Income Distribution')
plt.xscale('log')
plt.xlim(9999, 200000)  # Adjust the limits according to your desired range
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '${:,.0f}'.format(x)))
plt.savefig('Annual_Income_Distribution.png')
plt.show()

# Plot debt-to-income ratio distribution
plt.figure(figsize=(10, 7))
sns.histplot(df['dti'], kde=True)
plt.xlabel('Debt-to-Income Ratio')
plt.ylabel('Frequency')
plt.title('Debt-to-Income Ratio Distribution')
plt.savefig('Debt_to_Income_Ratio_Distribution.png')
plt.show()

# Plot loan purpose distribution
plt.figure(figsize=(10, 7))
sns.countplot(data=df_legacy, y='purpose', order=df_legacy['purpose'].value_counts().index, palette=colors)
plt.xlabel('Count')
plt.ylabel('Loan Purpose')
plt.title('Distribution of Loan Purposes')
plt.savefig('Loan_Purpose_Distribution.png')
plt.show()

# Plot loan status distribution (target variable)
plt.figure(figsize=(10, 7))
sns.countplot(data=df, x='loan_status')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.title('Distribution of Loan Status')
plt.xticks(rotation=45)
plt.savefig('Loan_Status_Distribution.png')
plt.show()

# Histogram
plt.figure(figsize=(10, 7))
plt.hist(df['loan_amnt'], bins=30, color=colors[0], edgecolor='white')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Loan Amounts')
plt.savefig('Loan_Amount_Distribution.png')
plt.show()

# Bar Chart
plt.figure(figsize=(10, 7))
df['term'].value_counts().sort_index().plot(kind='bar', color=[colors[0], colors[1]])
plt.xlabel('Loan Term')
plt.ylabel('Count')
plt.title('Distribution of Loan Terms')
plt.savefig('Loan_Term_Distribution.png')
plt.show()

# Pie Chart
plt.figure(figsize=(10, 7))
grade_counts = df_legacy['grade'].value_counts()
grade_order = sorted(grade_counts.index)
grade_counts.loc[grade_order].plot(kind='pie', autopct='%1.1f%%', colors=colors[2:])
plt.title('Distribution of Loan Grades')
plt.ylabel('')
plt.savefig('Loan_Grade_Distribution.png')
plt.show()

# Boxplot
plt.figure(figsize=(10, 7))
sns.boxplot(x=df_legacy['grade'], y=df_legacy['int_rate'],order=['A','B','C','D','E','F','G'])
plt.xlabel('Loan Grade')
plt.ylabel('Interest Rate')
plt.title('Interest Rate Distribution by Loan Grade')
plt.savefig('Interest_Rate_by_Loan_Grade.png')
plt.show()

# Scatterplot with plt.scatter
plt.figure(figsize=(10, 7))
plt.scatter(x=df['loan_amnt'], y=df['annual_inc'], alpha=0.1, color=colors[4], s=1)
plt.xlabel('Loan Amount')
plt.ylabel('Annual Income')
plt.title('Loan Amount vs. Annual Income')
plt.yscale('log')
median_annual_inc = df['annual_inc'].median()
plt.axhline(y=median_annual_inc, color='red', linestyle='--', label='Median Annual Income')
formatter = ScalarFormatter()
formatter.set_scientific(False)
plt.gca().yaxis.set_major_formatter(formatter)
plt.legend()
plt.savefig('Loan_Amount_vs_Annual_Income.png')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 7))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('Correlation_Heatmap.png')
plt.show()

# Stacked Bar Chart
plt.figure(figsize=(10, 7))
stacked_df_indv = df.groupby(['term', 'grade']).size().unstack()
stacked_df_indv.plot(kind='bar', stacked=True, color=colors)
plt.xlabel('Loan Term')
plt.ylabel('Count of Loans')
plt.title('Distribution of Loan Grades by Term')
plt.legend(title='Loan Grade')
plt.savefig('Loan_Grades_by_Term.png')
plt.show()
