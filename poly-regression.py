import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


df = pd.read_csv('data.csv')
x = df[['X1', 'X2']]
y = df['Y']
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(x, y)

test = pd.DataFrame(np.array([[1, 1]]), columns=x.columns)

print('predict', test.to_string(index=False, header=False), '=', model.predict(test))
print('score', model.score(x, y))
print(model.named_steps['polynomialfeatures'].get_feature_names_out(x.columns))
print(model.named_steps['linearregression'].coef_)
print('done')
