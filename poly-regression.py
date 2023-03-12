import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def read_csvs(files):
  """Read and concatenate multiple CSV files in one data frame."""
  return pd.concat([pd.read_csv(file) for file in files])

def equation(features, coefficients):
  """Format coefficient and features into polynomial equation."""
  if len(features) != len(coefficients): raise IndexError
  def format_exponential(feature):
    match = re.search('(.*)\^(\d+)$', feature)
    if match: return f'pow({match.groups()[0]}, {match.groups()[1]})'
    return feature
  def format_feature(feature):
    if feature == '1': return ''
    subfeatures = feature.split(' ')
    return ' * ' + ' * '.join([format_exponential(subfeature) for subfeature in subfeatures])
  features = map(format_feature, features)
  return ' + '.join([f'{i}{j}' for i, j in zip(coefficients, features) if i])

# Train

df = read_csvs([f'data/{i}.csv' for i in range(3)])
x = df[['tick', 'v1x']]
y = df['p2x']
model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model.fit(x, y)

# Test

df_test = pd.read_csv('data/3.csv')
x_test = df_test[x.columns]
y_test = df_test[y.name]
spot_test = pd.DataFrame(np.array([[160, 300]]), columns=x.columns)

# Print results
print('Predict', spot_test.to_string(index=False, header=False), '=', model.predict(spot_test))
print('Score train:', model.score(x, y))
print('Score test:', model.score(x_test, y_test))

# Print equation

features, coefficients = model.named_steps['polynomialfeatures'].get_feature_names_out(x.columns), model.named_steps['linearregression'].coef_
print('Features:', features)
print('Coefficients:', coefficients)
print(equation(features, coefficients))
print('Done')
