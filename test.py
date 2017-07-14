# coding: utf-8

import som
import os
import random
import json
import pandas as pd

# Hacarus のレシピデータを読み込む
recipe = pd.read_csv("data/recipe/recipe.csv")

# コンビニの食材データを読み込む
conveni = pd.read_csv("data/conveniencestore/conveni.csv")

# プロテインのデータを読み込む
protein = pd.read_csv("data/protein/protein.csv")

# データを結合
foods = pd.concat([recipe, conveni, protein], ignore_index=True)

# リコメンドに必要なデータを抽出
col_label = ['food_code', 'food_name', 'source_name', 'エネルギー', 'タンパク質', '脂質', '炭水化物', '食塩相当量', '野菜類',
             'ビタミンB1', 'ビタミンB2', 'ビタミンB6', 'ビタミンC', 'カルシウム', '鉄', '亜鉛']
extracted = pd.DataFrame()
for label in col_label:
    extracted[label] = foods[label]

features = extracted.iloc[:, 3:].as_matrix()

vs = []
for i in range(len(extracted.index)):
    vs.append(som.NamedVector(extracted.iloc[i, 1], extracted.iloc[i, 3:].as_matrix()))

random.shuffle(vs)

for v in vs:
    v.dump()

msom = som.SOM(vs)
#msom.arranged()
#msom.fission()
msom.execute_with_animation('result.txt')

#-----------
#while msom.update_som_assign():
#    pass
#msom.reassign_all()
#-----------

#-----------
#msom.update_som_repeatedly(200)
#msom.clustering()
#-----------

#msom.show()
#msom.animation()
#msom.dump()
#msom.write('result.txt')
