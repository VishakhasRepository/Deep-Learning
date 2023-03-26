import pandas as pd

# load the dataset
data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

# drop irrelevant columns
data.drop(['Rank', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1, inplace=True)

# group data by customer ID and game title
grouped_data = data.groupby(['CustomerID', 'Name'])['Global_Sales'].sum().reset_index()

# create pivot table with customer IDs as rows and games as columns
pivot_table = grouped_data.pivot(index='CustomerID', columns='Name', values='Global_Sales').fillna(0)

# convert pivot table to a list of lists
transactions = pivot_table.apply(lambda x: x.astype(bool).tolist(), axis=1).tolist()


from apyori import apriori

# run Apriori algorithm to generate association rules
rules = apriori(transactions, min_support=0.005, min_confidence=0.2, min_lift=3, max_length=2)

# convert association rules to a list
association_rules = list(rules)


# define function to convert association rules to a dataframe
def rule_to_dataframe(rule):
    antecedents = list(rule[0])
    consequents = list(rule[1])
    support = rule[2]
    confidence = rule[3]
    lift = rule[4]
    return pd.DataFrame({'antecedents': antecedents, 'consequents': consequents, 'support': support, 'confidence': confidence, 'lift': lift})

# convert association rules to a dataframe
rules_df = pd.concat([rule_to_dataframe(rule) for rule in association_rules], ignore_index=True)

# filter rules by lift, confidence, and support
filtered_rules = rules_df[(rules_df['lift'] >= 3) & (rules_df['confidence'] >= 0.2) & (rules_df['support'] >= 0.005)]

# sort rules by lift and confidence
sorted_rules = filtered_rules.sort_values(['lift', 'confidence'], ascending=False)
