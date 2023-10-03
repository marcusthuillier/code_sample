import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

class ModelTraining:
    def __init__(self, cfg: dict, train_data: pd.DataFrame, test_data: pd.DataFrame, we_have_actuals: bool = False) -> None:
        for k, v in cfg["dataset"].items():
            setattr(self, k, v)

        self.train_data = train_data
        self.test_data = test_data
        self.we_have_actuals = we_have_actuals

    def split_features_target(self) -> None:

        columns = self.num_features.copy()
        columns.extend(self.non_num_features)

        self.X_train = self.train_data[columns]
        self.X_test = self.test_data[columns]

        self.y_train = self.train_data[self.target]
        if self.we_have_actuals:
            self.y_test = self.test_data[self.target]

    def one_hot_encoding(self) -> None:
        '''
        One hot encoding for all string columns
        '''
        self.X_train["train"] = 0
        self.X_test["train"] = 1

        data = pd.concat([self.X_train,self.X_test])

        for column in self.non_num_features:
            data = pd.get_dummies(data, columns=[column], prefix=column)

        self.X_train = data[data["train"] == 0]
        self.X_test = data[data["train"] == 1]

    def initialize_model_parameters(self) -> GridSearchCV:
        '''' 
        intitiate GridSearch Object
        '''
        n_splits = 3

        params = {
        'max_depth': [2,6,10],  # Maximum depth of a tree
        'learning_rate': [0.1, 0.01],  # Step size shrinkage used in boosting
        'n_estimators': [50,100],  # Number of boosting stages
        'gamma': [0, 0.5]  # Minimum loss reduction required to make a further partition on a leaf node
        }

        gsc = GridSearchCV(
            estimator=XGBRegressor(random_state=0),
            param_grid=params,
            cv=n_splits, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            refit=True)
        
        return gsc

    def fitting_model(self) -> None:
        ''' 
        Fit inititated model
        '''
        self.xgb = self.initialize_model_parameters()
        self.xgb.fit(self.X_train, self.y_train)
        predictions = self.xgb.predict(self.X_train)

        print("Model Training Results")
        print("MAE",mean_absolute_error(self.y_train, predictions))
        print("R2",r2_score(self.y_train, predictions))
        
    def save_model_objects(self) -> None:
        ''' 
        Save model as pickle object
        '''
        pickle.dump(self.xgb, open(self.model_path, 'wb')) 

    def undumnify(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' 
        Restore columns
        '''
        for column in self.non_num_features:
            dummy_cols = [col for col in df.columns if column in col]
            df[column] = df[dummy_cols].idxmax(axis=1)
            df[column] = df[column].str.replace(column + "_","")
            df.drop(dummy_cols, axis=1, inplace=True)
        return df

    def make_predictions(self) -> None:
        ''' 
        Predict on test set
        '''
        self.X_test['predictions'] = self.predictions
        self.X_test = self.undumnify(self.X_test)
        self.X_test.to_csv(self.output_path, index=False)

    def test_accuracy(self) -> None:
        ''' 
        evaluate prediction on test set if we have the predictions
        '''
        self.predictions = self.xgb.predict(self.X_test)

        print("Model Testing Results")
        print("MAE",mean_absolute_error(self.y_test, self.predictions))
        print("R2",r2_score(self.y_test, self.predictions))    

    def train_model(self) -> None:
        self.split_features_target()
        self.one_hot_encoding()
        self.fitting_model()
        self.save_model_objects()
        if self.we_have_actuals:
            self.test_accuracy()
            self.make_predictions()

