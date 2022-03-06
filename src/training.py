import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)

def categorical_numeric_features(df, features):
    """
    Splits a list of features into categorical and numerical features
    
    """
    cat_feat=[]
    num_feat=[]
    for i in features:
        if df[i].dtype == 'object':
            cat_feat.append(i)
        else: num_feat.append(i)

    return cat_feat, num_feat

def run_model():
    
    logging.info("training started")
    
    df = pd.read_csv("../data/bank-full.csv", delimiter = ";")
    
    cols_to_keep = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
            'loan', 'contact', 'duration', 'campaign', 'previous', 'poutcome','y']
    
    features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
            'loan', 'contact', 'duration', 'campaign', 'previous', 'poutcome']
    
    df1 = df[cols_to_keep]
    
    test_df = df1.sample(frac=0.1, random_state=99)
    
    # Build model

    y = df1.pop('y')
    X = df1

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=99)
    
    cat_feat, num_feat = categorical_numeric_features(df1, features)
    
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = OneHotEncoder(handle_unknown = 'ignore')
    
    preprocessor = ColumnTransformer(transformers =[
    ("num", numeric_transformer, num_feat),
    ("cat", categorical_transformer, cat_feat)
    ])
    
    clf = Pipeline(steps=[("preprocessor", preprocessor),
                     ("classifier", LogisticRegression())])

    clf.fit(X_train, y_train)

    logging.info("model_score: {:.3f}".format(clf.score(X_val, y_val)))
    
    logging.info("training completed")
    print("training completed")
    

if __name__ == '__main__':
    run_model()
