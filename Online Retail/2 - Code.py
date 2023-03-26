import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_excel('Online Retail.xlsx')

# Dropping rows with missing values
df.dropna(inplace=True)

# Removing unnecessary characters from the StockCode
df['StockCode'] = df['StockCode'].str.strip()

# Filtering out cancelled transactions
df = df[~df['InvoiceNo'].str.contains('C')]

# Creating the basket for each transaction
basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# Applying the Apriori algorithm
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)

# Generating the association rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

# Sorting the rules by lift in descending order
rules.sort_values('lift', ascending=False, inplace=True)

# Displaying the top 10 rules
print(rules.head(10))
