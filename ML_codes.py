#!/usr/bin/env python
# coding: utf-8

# # 1. One Hot Encoding method 1



from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ['Make', 'Colour', 'Doors']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([('one_hot', one_hot, categorical_features)],
                               remainder='passthrough')
transformed_X = transformer.fit_transform(X)


# # 2. One Hot Encoding method 2



dummies = pd.get_dummies(car_sales[['Make', 'Colour', 'Doors']])


# # Fill NA method 1



car_sales_missing['Make'].fillna('missing', inplace = True)
car_sales_missing['Colour'].fillna('missing', inplace = True)
car_sales_missing['Odometer (KM)'].fillna(car_sales_missing['Odometer (KM)'].mean(), inplace = True)
car_sales_missing['Doors'].fillna(4, inplace = True)


# # Fill NA method 2



from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

cat_imputer = SimpleImputer(strategy = 'constant', fill_value = 'missing')
door_imputer = SimpleImputer(strategy = 'constant', fill_value = 4)
num_imputer = SimpleImputer(strategy = 'mean')

cat_features = ['Make', 'Colour']
door_feature = ['Doors']
num_features = ['Odometer (KM)']

imputer = ColumnTransformer([('cat_imputer', cat_imputer, cat_features), 
                            ('door_imputer', door_imputer, door_feature),
                            ('num_feature', num_imputer, num_features)])

filled_X = imputer.fit_transform(X)


# # Validation Score



from sklearn.model_selection import cross_val_score

cross_val_score(clf, X, y)


# # Randomized Search CV



from sklearn.model_selection import RandomizedSearchCV

grid = {'n_estimators': [10, 100, 200, 500, 1000, 1200],
        'max_depth': [None, 5, 10, 20, 30],
        'max_features': ['auto', 'sqrt'],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 4]
       }
np.random.seed(42)

X = heart_disease_shuffled.drop('target', axis = 1)
y = heart_disease_shuffled.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
clf = RandomForestClassifier(n_jobs = 1)

rs_clf = RandomizedSearchCV(estimator = clf, param_distributions=grid,
                           n_iter=10, cv = 5, verbose = 2)
rs_clf.fit(X_train, y_train)




rs_clf.best_params_


# # Grid Search CV



from sklearn.model_selection import GridSearchCV, train_test_split

np.random.seed(42)

X = heart_disease_shuffled.drop('target', axis = 1)
y = heart_disease_shuffled.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
clf = RandomForestClassifier(n_jobs = 1)

gs_clf = GridSearchCV(estimator = clf, param_grid=grid_2, cv = 5, verbose = 2)
gs_clf.fit(X_train, y_train)




gs_clf.best_params_

