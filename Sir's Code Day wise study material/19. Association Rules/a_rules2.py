import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

groceries = []
with open("/home/dai/33/machine_learning/Datasets/Groceries.csv","r") as f:
    groceries = f.read()
    groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
    
te = TransactionEncoder()
te_ary = te.fit(groceries_list).transform(groceries_list)
print(te_ary)
fp_df = pd.DataFrame(te_ary, columns=te.columns_)
print(fp_df)
# one_freq = fp_df.sum().reset_index()
# one_freq.columns=['Items', 'Freq']
# print(one_freq.sort_values(by='Freq', ascending=False))

# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.01, use_colnames=True)
# and convert into rules
rules = association_rules(itemsets, metric='confidence', min_threshold=0.01)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
rules = rules[rules['lift']>1]
rules.sort_values(by='lift', ascending=False)
print(rules)
