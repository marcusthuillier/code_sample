import pandas as pd
import numpy as np

class DataIngestion:
    def __init__(self, cfg: dict) -> None:
        for k, v in cfg["dataset"].items():
            setattr(self, k, v)

        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.data = pd.DataFrame()

    def read_train_dataset(self) -> None:
        '''
        Read in the training dataset
        '''
        try:
            self.train_data = pd.read_csv(self.training_path)
        except FileNotFoundError:
            pass

    def read_test_dataset(self) -> None:
        '''
        Read in the testing dataset
        '''
        try:
            self.test_data = pd.read_csv(self.testing_path)
        except FileNotFoundError:
            pass

    def consolidate_dataset(self) -> None:
        '''
        Consolidates train and test data
        '''

        self.train_data["train"] = 1
        self.test_data["train"] = 0
        self.data = pd.concat([self.train_data, self.test_data])

    def data_check(self) -> None:
        '''
        Making sure all data is in correct format
        '''

        for column in self.numerical_columns:
            self.data[column] = self.data[column].astype(int)
        
        self.data[self.date_column] = pd.to_datetime(self.data[self.date_column], errors='coerce')

        for column in self.string_columns:
            self.data[column] = self.data[column].str.lower()

    def metadata_column_creation(self) -> None:
        '''
        Extra Metadata from existing information
        '''
        world_cup_matches = "rugby world cup"
        tier_one_teams = self.tier_one_teams

        self.data["home_spread"] = self.data["home_score"] - self.data["away_score"]
        self.data["true_home_adv"] = np.where(self.data["home_team"] == self.data["country"], 1, 0)
        self.data["world_cup_game"] = np.where(self.data["competition"].str.contains(world_cup_matches, regex= True), 1, 0)

        two_tier_one_teams = self.data["home_team"].str.contains(r'\b(?:{})\b'.format('|'.join(tier_one_teams))) & self.data["away_team"].str.contains(r'\b(?:{})\b'.format('|'.join(tier_one_teams))) 
        self.data["tier_one_matches"] = np.where(two_tier_one_teams, 1, 0)
        home_tier_one_team = self.data["home_team"].str.contains(r'\b(?:{})\b'.format('|'.join(tier_one_teams))) & ~self.data["away_team"].str.contains(r'\b(?:{})\b'.format('|'.join(tier_one_teams))) 
        self.data["home_tier_one_matches"] = np.where(home_tier_one_team, 1, 0)
        away_tier_one_team = ~self.data["home_team"].str.contains(r'\b(?:{})\b'.format('|'.join(tier_one_teams))) & self.data["away_team"].str.contains(r'\b(?:{})\b'.format('|'.join(tier_one_teams))) 
        self.data["away_tier_one_matches"] = np.where(away_tier_one_team, 1, 0)

    def offensive_stats_columns(self) -> None:
        ''' 
        Average out offensive stats
        '''
        self.data.sort_values(by=self.date_column, inplace=True)

        self.data["home_points_scored_avg"] = self.data.groupby('home_team')['home_score'].transform(lambda x: x.rolling(5, 1).mean())
        self.data["away_points_scored_avg"] = self.data.groupby('away_team')['away_score'].transform(lambda x: x.rolling(5, 1).mean())

    def defensive_stats_columns(self) -> None:
        ''' 
        Average out defensive stats
        '''
        self.data.sort_values(by=self.date_column, inplace=True)

        self.data["home_points_conceded_avg"] = self.data.groupby('home_team')['away_score'].transform(lambda x: x.rolling(5, 1).mean())
        self.data["away_points_conceded_avg"] = self.data.groupby('away_team')['home_score'].transform(lambda x: x.rolling(5, 1).mean())

    def unpack_train_test(self) -> tuple:
        ''' 
        Unpack dataset into three objects for models
        '''
        self.train_data = self.data[self.data["train"] == 1]
        self.test_data = self.data[self.data["train"] == 0]
    
    def finalize_dataset(self) -> None:
        '''
        Create take the latest output data after setup, filter it on the prediction date, merge with MAD and item information and transform to create model input file
        '''
        self.read_train_dataset()
        self.read_test_dataset()
        self.consolidate_dataset()
        self.data_check()
        self.metadata_column_creation()
        self.offensive_stats_columns()
        self.defensive_stats_columns()
        self.unpack_train_test()

        return self.train_data, self.test_data