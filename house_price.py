import pandas
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import math

#функция ошибки, по которой будем оценивать качество работы метода
def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 
                 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

# Чтение данных
df_train = pandas.read_csv('data/train.csv', index_col='Id')
df_test = pandas.read_csv('data/test.csv', index_col='Id')
df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

#Закодируем категориальные переменные
df_train = pandas.get_dummies(df_train, columns=['Neighborhood'])
df_train = pandas.get_dummies(df_train, columns=['GarageType'])
df_train = pandas.get_dummies(df_train, columns=['SaleType'])
df_train = pandas.get_dummies(df_train, columns=['MSZoning'])
df_train['HouseAge'] = df_train['YrSold'] - df_train['YearBuilt']

#составим список фич для обучения
features = ['OverallQual', 'OverallCond', 'GrLivArea', 
            'LotArea', 'GarageCars', 'GarageArea', 'YearBuilt', 
            'YearRemodAdd', 'TotRmsAbvGrd', 'Fireplaces', 'MoSold', 
            'HouseAge']
features += list(filter(lambda x : x.startswith('Neighborhood'), list(df_train.columns)))
features += list(filter(lambda x : x.startswith('GarageType'), list(df_train.columns)))
features += list(filter(lambda x : x.startswith('SaleType'), list(df_train.columns)))
features += list(filter(lambda x : x.startswith('MSZoning'), list(df_train.columns)))

#разобъем выборку на обучающую и тестовую, по которой будем мерить точность
X_train, X_test, y_train, y_test = train_test_split(df_train[features], 
                                                    df_train['SalePrice'], 
                                                    train_size = 0.8, 
                                                    random_state=42)

rf = Ridge(random_state=42)
rf.fit(X_train, y_train)

fi=list(zip(features, rf.coef_))
fi=pandas.DataFrame(rf.feature_importances_, features)
print(fi)

df_train['Prediction'] = rf.predict(df_train[features])
df_train['AbsErr'] = abs(df_train['SalePrice'] - df_train['Prediction'])

#выведем точность на тестовой части метрики
print(rmsle(list(rf.predict(X_test)), list(y_test)))


#Закодируем категориальные переменные
df_test = pandas.get_dummies(df_test, columns=['Neighborhood'])
df_test = pandas.get_dummies(df_test, columns=['GarageType'])
df_test = pandas.get_dummies(df_test, columns=['SaleType'])
df_test = pandas.get_dummies(df_test, columns=['MSZoning'])
df_test['HouseAge'] = df_test['YrSold'] - df_test['YearBuilt']

for f in set(features) - set(df_train.columns):
    df_test[f] = 0

df_test['SalePrice'] = rf.predict(df_test[features])


df_test['SalePrice'].to_csv('submission.csv', header=1)

