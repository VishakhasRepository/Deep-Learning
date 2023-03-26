import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
df = pd.read_csv('newsCorpora.csv', sep='\t', header=None)
df.columns = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']

# Preprocess and clean the data
df = df[['ID', 'TITLE', 'PUBLISHER', 'CATEGORY']]
df = df[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])]
df['TITLE'] = df['TITLE'].str.lower()
df['TITLE'] = df['TITLE'].str.replace('[^\w\s]','')
df['TITLE'] = df['TITLE'].str.strip()
df = df.drop_duplicates()

# Convert the data into a transaction format
df['ARTICLES'] = df.groupby('ID')['TITLE'].transform(lambda x: ','.join(x))
df = df[['ID', 'ARTICLES']].drop_duplicates()
df = df.dropna()

# Apply association rule mining
te = TransactionEncoder()
te_ary = te.fit_transform(df['ARTICLES'].str.split(','))
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Visualize the results
rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
rules.head()

# Provide recommendations
# TODO

# Write a report
# TODO
