# import all libraries needed
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# the custom scaler class 
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.columns = columns
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# create the special class that we are going to use from here on to predict new data
class absenteeism_model():
      
        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler' files which were saved
            with open('model','rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
        
        # take a data file (*.csv) and preprocess it in the same way as in the lectures
        def load_and_clean_data(self, data_file):
            
            # import the data
            df = pd.read_csv(data_file,delimiter=',')

            df.columns = (df.columns
                          .str.strip() #remove white space from head & tail
                          .str.replace(' ', '_') #replace whitespace with _
                          .str.lower()) #lowercase the columns name
            
            # store the data in a new variable for later use
            self.df_with_predictions = df.copy()

            # drop the 'ID' column
            df = df.drop(['id'], axis = 1)

            # to preserve the code we've created in the previous section, we will add a column with 'NaN' strings
            df['absenteeism_time_in_hours'] = 'NaN'

            # create a separate dataframe, containing dummy values for ALL avaiable reasons
            reason_columns = pd.get_dummies(df['reason_for_absence'], drop_first = True, dtype=int)
            
            # split reason_columns into 4 types
            reason_type_1 = reason_columns.loc[:,1:14].any(axis=1).astype(int)
            reason_type_2 = reason_columns.loc[:,15:17].any(axis=1).astype(int)
            reason_type_3 = reason_columns.loc[:,18:21].any(axis=1).astype(int)
            reason_type_4 = reason_columns.loc[:,22:].any(axis=1).astype(int)
            
            # to avoid multicollinearity, drop the 'Reason for Absence' column from df
            df = df.drop(['reason_for_absence'], axis = 1)
            
            # concatenate df and the 4 types of reason for absence
            df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)
            
            # assign names to the 4 reason type columns           
            column_names = ['date', 'transportation_expense', 'distance_to_work', 'age',
                           'daily_work_load_average', 'body_mass_index', 'education', 'children',
                           'pets', 'absenteeism_time_in_hours', 'reason_1', 'reason_2', 'reason_3', 'reason_4']
            df.columns = column_names

            # ensure reason columns are integers
            df['reason_1'] = df['reason_1'].astype(int)
            df['reason_2'] = df['reason_2'].astype(int)
            df['reason_3'] = df['reason_3'].astype(int)
            df['reason_4'] = df['reason_4'].astype(int)

            # re-order the columns in df
            column_names_reordered = ['reason_1', 'reason_2', 'reason_3', 'reason_4', 'date', 'transportation_expense', 
                                      'distance_to_work', 'age', 'daily_work_load_average', 'body_mass_index', 'education', 
                                      'children', 'pets', 'absenteeism_time_in_hours']
            df = df[column_names_reordered]
      
            # convert the 'Date' column into datetime
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

            # create a list with month values retrieved from the 'Date' column
            list_months = []
            for i in range(df.shape[0]):
                list_months.append(df['date'][i].month)

            # insert the values in a new column in df, called 'Month Value'
            df['month_value'] = list_months

            # create a new feature called 'Day of the Week'
            df['day_of_the_week'] = df['date'].apply(lambda x: x.weekday())


            # drop the 'Date' column from df
            df = df.drop(['date'], axis = 1)

            # re-order the columns in df
            column_names_upd = ['reason_1', 'reason_2', 'reason_3', 'reason_4', 'month_value', 'day_of_the_week',
                                'transportation_expense', 'distance_to_work', 'age',
                                'daily_work_load_average', 'body_mass_index', 'education', 'children',
                                'pets', 'absenteeism_time_in_hours']
            df = df[column_names_upd]

            # map 'Education' variables; the result is a dummy
            df['education'] = df['education'].map({1:0, 2:1, 3:1, 4:1})

            # replace the NaN values
            df = df.fillna(value=0)

            # drop the original absenteeism time
            df = df.drop(['absenteeism_time_in_hours'],axis=1)
            
            # drop the variables we decide we don't need
            df = df.drop(['day_of_the_week','daily_work_load_average','distance_to_work'],axis=1)
            
            # reorder columns to match training data order
            df = df[['reason_1', 'reason_2', 'reason_3', 'reason_4', 'month_value', 'transportation_expense', 'age', 'body_mass_index', 'education', 'children', 'pets']]
            
            # we have included this line of code if you want to call the 'preprocessed data'
            self.preprocessed_data = df.copy()
            
            # we need this line so we can use it in the next functions
            self.data = self.scaler.transform(df).values
    
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
        
        # a function which outputs 0 or 1 based on our model
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs
        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
                return self.preprocessed_data