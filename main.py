import pandas as pd
import chardet
from mlxtend.frequent_patterns import apriori, association_rules

# Определение кодировки файла
with open('./Transactions5.txt', 'rb') as f:
    result = chardet.detect(f.read())

# Загрузка данных из файла с указанием правильной кодировки
data = pd.read_csv('./Transactions5.txt', sep=';', encoding=result['encoding'])

# Преобразование данных в формат, подходящий для анализа ассоциативных правил
basket = (data
          .groupby(['НомерЧека', 'Наименование'])['Количество']
          .sum().unstack().reset_index().fillna(0)
          .set_index('НомерЧека'))

# Кодирование данных для использования алгоритма Apriori
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

# Поиск частых наборов элементов с использованием алгоритма Apriori
frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)

# Генерация ассоциативных правил
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print(rules)
