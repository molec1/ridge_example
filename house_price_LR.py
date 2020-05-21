import pandas
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import math
from sklearn.metrics import r2_score,mean_squared_error

#функция ошибки, по которой будем оценивать качество работы метода
def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 
                 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

#создадим функцию, в которую поместим все преобразования над датасетом
#чтобы делать их одинаково и над тестовой и над обучающим датасетами
def prepare_df(df):
    #закодируем все категориальные переменные как множество бинарных столбцов
    df = pandas.get_dummies(df, columns=['Neighborhood'])
    df = pandas.get_dummies(df, columns=['GarageType'])
    df = pandas.get_dummies(df, columns=['SaleType'])
    df = pandas.get_dummies(df, columns=['MSZoning'])
    df = pandas.get_dummies(df, columns=['OverallQual'])
    df = pandas.get_dummies(df, columns=['OverallCond'])
    df = pandas.get_dummies(df, columns=['BsmtFinType1'])
    df = pandas.get_dummies(df, columns=['BsmtFinType2'])
    df = pandas.get_dummies(df, columns=['SaleCondition'])
    df = pandas.get_dummies(df, columns=['BldgType'])
    df = pandas.get_dummies(df, columns=['HouseStyle'])
    
    #нормируем наши площади, чтобы среднее значение было ~1
    df['LotArea'] = df['LotArea'] / 10000
    df['GrLivArea'] = df['GrLivArea'] / 1500
    df['BsmtFinSF1'] = df['BsmtFinSF1'] / 400
    df['BsmtFinSF2'] = df['BsmtFinSF2'] / 40
    df['BsmtUnfSF'] = df['BsmtUnfSF'] / 500
    
    #корни из основных площадей
    df['sqLotArea'] = df['LotArea']**0.5
    df['sqGrLivArea'] = df['GrLivArea']**0.5
    
    df['YearRemodAdd5'] = df['YearRemodAdd']//10*10
    df = pandas.get_dummies(df, columns=['YearRemodAdd5'])
    
    #поскольку район в основном определяет стоимость земли, посчитаем
    #аналог стоимости квадрата для каждого района
    for f in list(filter(lambda x : x.startswith('Neighborhood'), list(df.columns))):
        df[f] = df[f] * df['LotArea']
    #далее попытаемся посчитать стоимость квадрата для дома
    #она будет состоять из нескольких слагаемых
    for f in list(filter(lambda x : x.startswith('MSZoning'), list(df.columns))):
        df[f] = df[f] * df['GrLivArea']
    for f in list(filter(lambda x : x.startswith('BldgType'), list(df.columns))):
        df[f] = df[f] * df['GrLivArea']
    for f in list(filter(lambda x : x.startswith('HouseStyle'), list(df.columns))):
        df[f] = df[f] * df['GrLivArea']
    for f in list(filter(lambda x : x.startswith('OverallQual'), list(df.columns))):
        df[f] = df[f] * df['GrLivArea']
    for f in list(filter(lambda x : x.startswith('OverallCond'), list(df.columns))):
        df[f] = df[f] * df['GrLivArea']
    for f in list(filter(lambda x : x.startswith('YearRemodAdd5'), list(df.columns))):
        df[f] = df[f] * df['GrLivArea']
    #также прикинем стоимость одного паркоместа, учитывая его тип
    for f in list(filter(lambda x : x.startswith('GarageType'), list(df.columns))):
        df[f] = df[f] * df['GarageCars']
    #подвал. его тип и площади
    for f in list(filter(lambda x : x.startswith('BsmtFinType1'), list(df.columns))):
        df[f] = df[f] * df['BsmtFinSF1']
    for f in list(filter(lambda x : x.startswith('BsmtFinType2'), list(df.columns))):
        df[f] = df[f] * df['BsmtFinSF2']
    
    #добавим несколько бинарных переменных. Есть ли в доме бассейн,гараж и тп
    df['ExistsPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df['ExistsGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    df['ExistsFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    df['ExistsBasement'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    df['Exists2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    
    #и 2 сдизайнеренных фичи. Первая - что-то типа процента оставшегося износа дома. 
    #Предполагаем, что дома дольше 100 лет не стоят. Вторая - норм. сумма всех крылец. Крытых и не крытых
    df['RestPercent'] = 1 - (df['YrSold'] - df['YearBuilt'])*1.0/100        
    df['AllPorch'] = (df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'])/100
    return df
   
# Чтение данных
df_train = pandas.read_csv('data/train.csv', index_col='Id')
df_test = pandas.read_csv('data/test.csv', index_col='Id')
#к счастью, пропусков в важных данных почти нет и можно просто забивать 0.
df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

#удалим выбросы. В статистике присутствуют слишком большие либо слишком маленькие
#объекты, которые только мешают анализировать данные. Оставим серединку
df_train = df_train[(df_train['SalePrice']<600000) &
        (df_train['GrLivArea']<4000) &
        (df_train['LotArea']<50000) &
        (df_train['SalePrice']>40000) & 
        (df_train['GrLivArea']>350) &
        (df_train['LotArea']>3000) ].copy()

#создаем копию исходного датасета для дальнейшего анализа
df_rep = df_train.copy()
#определим все переменные
df_train = prepare_df(df_train)

#составим список фич для обучения
features = [ 'BsmtUnfSF', 'AllPorch',            
            'sqLotArea', 'sqGrLivArea',
            'ExistsPool', 'ExistsGarage', 'ExistsFireplace', 
            'Exists2ndFloor', 'ExistsBasement', 
            'RestPercent']

features += list(filter(lambda x : x.startswith('Neighborhood'), list(df_train.columns)))
features += list(filter(lambda x : x.startswith('GarageType'), list(df_train.columns)))
features += list(filter(lambda x : x.startswith('MSZoning'), list(df_train.columns)))
features += list(filter(lambda x : x.startswith('BldgType'), list(df_train.columns)))
features += list(filter(lambda x : x.startswith('OverallQual'), list(df_train.columns)))
features += list(filter(lambda x : x.startswith('OverallCond'), list(df_train.columns)))
features += list(filter(lambda x : x.startswith('YearRemodAdd5'), list(df_train.columns)))
features += list(filter(lambda x : x.startswith('BsmtFinType1'), list(df_train.columns)))
features += list(filter(lambda x : x.startswith('BsmtFinType2'), list(df_train.columns)))
features += list(filter(lambda x : x.startswith('SaleCondition'), list(df_train.columns)))

#разобъем выборку на обучающую и тестовую, по которой будем мерить точность
X_train, X_test, y_train, y_test = train_test_split(df_train[features], 
                                                    df_train['SalePrice'], 
                                                    train_size = 0.8, 
                                                    random_state=42)
#создадим регрессию и обучим ее
rf = Ridge(alpha=2, fit_intercept=0, random_state=42)
rf.fit(X_train, y_train)

#выведем ошибку и точность на тестовой части метрики
print('train rmsle:', rmsle(list(rf.predict(X_train)), list(y_train)))
print('test  rmsle:', rmsle(list(rf.predict(X_test)), list(y_test)))
print('train r2:  ', r2_score(list(rf.predict(X_train)), list(y_train)))
print('test  r2:  ', r2_score(list(rf.predict(X_test)), list(y_test)))

#и посчитаем, собственно, стоимость для объектов с неизвестной ценой
X_late_test = prepare_df(df_test)

for f in set(features) - set(X_late_test.columns):
    X_late_test[f] = 0

#переобучим модель на полном доступном датасете
rf.fit(df_train[features], df_train['SalePrice'])
#зачем мы здесь и собрались - эти коэфициенты при переменных модели
#мы можем перенести в любое приложение и использовать для оценки стоимости объекта
features_coeffs = pandas.DataFrame(rf.coef_, features)

#посчитаем оценку для всей выборки попробуем оценить корреляцию
#остатков с переменными. Эту информацию можно использовать для поиска новых фич
df_rep['Prediction'] = rf.predict(df_train[features])
df_rep['Err'] = df_rep['SalePrice'] - df_rep['Prediction']
df_rep['AbsErr'] = abs(df_rep['SalePrice'] - df_rep['Prediction'])/df_rep['SalePrice']
features_correlation = df_rep.corr()[['Err', 'AbsErr']]

X_late_test['SalePrice'] = rf.predict(X_late_test[features])
#сохраним результат
X_late_test['SalePrice'].to_csv('submission.csv', header=1)

