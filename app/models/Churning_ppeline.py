import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import catboost as catb
from sklearn.metrics import precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import dill

df = pd.read_csv('BankChurners.csv')
df=df.drop(['CLIENTNUM'], axis=1)
Attrition_Flag_to_numbers = {'Existing Customer': 0, 'Attrited Customer': 1}
df['Attrition_Flag'] = df['Attrition_Flag'].map(Attrition_Flag_to_numbers)
x_RAW = df[df.columns[1:]]
y_RAW = df['Attrition_Flag']
X_train, X_test, y_train, y_test = train_test_split(x_RAW,y_RAW,test_size = 0.3,
                                                                 random_state =42,
                                                                 stratify=y_RAW)
features = ['Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Trans_Ct_log',
 'Percent_Total_Amt_Chng', 'Max_Trans', 'Total_Relationship_Count',
 'Total_Amt_Chng_Q4_Q1', 'Months_Inactive_12_mon', 'Total_Revolving_Bal']
target = 'Attrition_Flag'

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]

class DoubleLogTransformer(BaseEstimator, TransformerMixin):
    """Apply given transformation."""
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[f"{self.key}_log"] = np.log(np.log(X[self.key]))
        return X


class DivisionTransformer(BaseEstimator, TransformerMixin):
    """Apply given transformation."""

    def __init__(self, key1, key2):
        self.key1 = key1
        self.key2 = key2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x1 = self._get_selection1(X)
        x2 = self._get_selection2(X)
        X[f"{self.key1}_{self.key2}"] = x1 / x2
        return X

    def _get_selection1(self, X):
        assert isinstance(X, pd.DataFrame)
        return X[self.key1]

    def _get_selection2(self, X):
        assert isinstance(X, pd.DataFrame)
        return X[self.key2]

Total_Trans_Ct_log = Pipeline([
                ('DoubleLogTransformer', DoubleLogTransformer(key = 'Total_Trans_Ct'))
            ])
Percent_Total_Amt_Chng = Pipeline([
                ('DivisionTransformer', DivisionTransformer(key1='Total_Ct_Chng_Q4_Q1', key2 = 'Total_Trans_Ct'))
            ])
Mean_Trans = Pipeline([
                ('DivisionTransformer', DivisionTransformer(key1='Total_Trans_Amt', key2='Total_Trans_Ct'))
            ])
Max_Trans = Pipeline([
                ('DivisionTransformer', DivisionTransformer(key1='Total_Revolving_Bal', key2='Total_Trans_Amt_Total_Trans_Ct'))
            ])
Selected_Columns = Pipeline([
                ('selector', ColumnSelector(['Total_Trans_Amt', 'Total_Trans_Ct',
                'Total_Relationship_Count', 'Total_Amt_Chng_Q4_Q1',
                'Months_Inactive_12_mon', 'Total_Revolving_Bal',
                'Total_Trans_Ct_log', 'Total_Ct_Chng_Q4_Q1_Total_Trans_Ct',
                'Total_Revolving_Bal_Total_Trans_Amt_Total_Trans_Ct']))
            ])

pipeline = Pipeline([
    ('Mean_Trans', Mean_Trans),
    ('Max_Trans', Max_Trans),
    ('Total_Trans_Ct_log', Total_Trans_Ct_log),
    ('Percent_Total_Amt_Chng', Percent_Total_Amt_Chng),
    ('Selected_Columns', Selected_Columns),
    ('classifier', catb.CatBoostClassifier(class_weights =[1, 5],
                            silent = True,
                            random_state = 21,
                            #cat_features = ['Total_Relationship_Count'
                             #              , 'Months_Inactive_12_mon'],
                            eval_metric = 'Precision',
                            early_stopping_rounds = 40))
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict_proba(X_test)[:, 1]

with open("Churning_pipeline.dill", "wb") as f:
    dill.dump(pipeline, f)
