#!/usr/bin/env python
#!-*-coding:utf-8 -*-
#!@Author:xugao
#         ┌─┐       ┌─┐
#      ┌──┘ ┴───────┘ ┴──┐
#      │                 │
#      │                 │
#      │    ＞  　　＜    │
#      │                 │
#      │  ....　⌒　....　│
#      │                 │
#      └───┐         ┌───┘
#          │         │
#          │         │
#          │         │
#          │         └──────────────┐
#          │                        │
#          │                        ├─┐
#          │                        ┌─┘
#          │                        │
#          └─┐  ┐  ┌───────┬──┐  ┌──┘
#            │ ─┤ ─┤       │ ─┤ ─┤
#            └──┴──┘       └──┴──┘
#                神兽保佑
#                BUG是不可能有BUG的!
import numpy
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

data = read_csv("demo.csv")
# print(data.sell)
plt.scatter(data.AD,data.sell)
print(data.corr())
plt.show()
demoModel = LinearRegression()
x = data[['AD']]
y = data[['sell']]
demoModel.fit(x,y)
print(demoModel.score(x,y))


alpha = demoModel.intercept_[0]
beta = demoModel.coef_[0][0]

print(alpha + beta*numpy.array([60,70]))
print(demoModel.predict([[60],[70]]))