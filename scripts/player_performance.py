import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#Ignoring XGBoost warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#Ignoring SciKit-Learn warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

def gridsearchcv_results(results):
    arg, params, flag = None, None, None
    split0 = results['split0_test_score'].max()
    split1 = results['split1_test_score'].max()
    if split0 >= split1:
        arg = results['split0_test_score'].argmax()
        flag = split0
    else:
        arg = results['split1_test_score'].argmax()
        flag = split1
    params = results['params'][arg]
    return flag, params

def player_performance(param,player_name,opposition=None,venue=None):

    res = {}

    #Extracting Targets and Features
    if param == 1:
        overall_batsman_details = pd.read_excel('./player_details/overall_batsman_details.xlsx', header=0)
        match_batsman_details = pd.read_excel('./player_details/match_batsman_details.xlsx',header=0)
        match_batsman_details.loc[:, 'date'].ffill(inplace=True)
        bat_match_details = match_batsman_details[match_batsman_details['name']==player_name]
        bat_match_details = bat_match_details[bat_match_details['opposition']==opposition]
        bat_overall_details = overall_batsman_details[overall_batsman_details['player_name'] == player_name][['player_name', 'team', 'innings', 'runs', 'average', 'strike_rate', 'centuries', 'fifties', 'zeros']]
        bat_features = bat_match_details.loc[:,['opposition', 'venue', 'innings_played','previous_average', 'previous_strike_rate', 'previous_centuries','previous_fifties', 'previous_zeros']]
        bat_targets = bat_match_details.loc[:,['runs']]
            
    elif param == 2:
        overall_bowler_details = pd.read_excel('./player_details/overall_bowler_details.xlsx',header=0)
        match_bowler_details = pd.read_excel('./player_details/match_bowler_details.xlsx',header=0)
        match_bowler_details.loc[:, 'date'].ffill(inplace=True)
        bowl_match_details = match_bowler_details[match_bowler_details['name']==player_name]
        bowl_match_details = bowl_match_details[bowl_match_details['opposition'] == opposition]
        bowl_overall_details = overall_bowler_details[overall_bowler_details['player_name']==player_name][['player_name','team','innings','wickets','average','strike_rate','economy','wicket_hauls']]
        bowl_features = bowl_match_details.loc[:,['opposition', 'venue', 'innings_played','previous_average', 'previous_strike_rate', 'previous_economy','previous_wicket_hauls']]
        bowl_targets = bowl_match_details.loc[:,['wickets']]

    elif param == 3:
        overall_batsman_details = pd.read_excel('./player_details/overall_batsman_details.xlsx', header=0)
        match_batsman_details = pd.read_excel('./player_details/match_batsman_details.xlsx',header=0)
        match_batsman_details.loc[:, 'date'].ffill(inplace=True)
        bat_match_details = match_batsman_details[match_batsman_details['name']==player_name]
        bat_match_details = bat_match_details[bat_match_details['opposition'] == opposition]
        bat_overall_details = overall_batsman_details[overall_batsman_details['player_name']==player_name][['player_name','team','innings','runs','average','strike_rate','centuries','fifties','zeros']]
        bat_features = bat_match_details.loc[:,['opposition', 'venue', 'innings_played','previous_average', 'previous_strike_rate', 'previous_centuries','previous_fifties', 'previous_zeros']]
        bat_targets = bat_match_details.loc[:,['runs']]

        overall_bowler_details = pd.read_excel('./player_details/overall_bowler_details.xlsx',header=0)
        match_bowler_details = pd.read_excel('./player_details/match_bowler_details.xlsx',header=0)
        match_bowler_details.loc[:, 'date'].ffill(inplace=True)
        bowl_match_details = match_bowler_details[match_bowler_details['name'] == player_name]
        bowl_match_details = bowl_match_details[bowl_match_details['opposition'] == opposition]
        bowl_overall_details = overall_bowler_details[overall_bowler_details['player_name']==player_name][['player_name','team','innings','wickets','average','strike_rate','economy','wicket_hauls']]
        bowl_features = bowl_match_details.loc[:,['opposition', 'venue', 'innings_played','previous_average', 'previous_strike_rate', 'previous_economy','previous_wicket_hauls']]
        bowl_targets = bowl_match_details.loc[:,['wickets']]

    #Pre_Processing
    le = preprocessing.LabelEncoder()
    sc = StandardScaler()

    #BatsmanPrediction
    if (param == 1 or param == 3):
        
        #Categorizing Runs
        bins = [0,10,30,50,80,120,250]
        labels = ["0","1","2","3","4","5"]
        bat_targets = pd.cut(bat_targets['runs'],bins,labels=labels,include_lowest=True)
        
        #Classification classes
        classes_bat = len(bat_targets.unique())

        if classes_bat >= 2:

            #Categorizing Opposition and Venue
            le.fit(bat_features.loc[:,'opposition'])
            opp_bat = le.transform([opposition])
            bat_features.loc[:,'opposition'] = le.transform(bat_features.loc[:,'opposition'])
            le.fit(bat_features.loc[:,'venue'])
            ven_bat = le.transform([venue])
            bat_features.loc[:,'venue'] = le.transform(bat_features.loc[:,'venue'])

            predict_bat = bat_overall_details[['innings','average','strike_rate','centuries','fifties','zeros']].values[0]

            #Scaling Non-Categorical Features
            bat_means = bat_features.loc[:,['innings_played','previous_average','previous_strike_rate','previous_centuries','previous_fifties','previous_zeros']].mean()
            bat_std = bat_features.loc[:,['innings_played','previous_average','previous_strike_rate','previous_centuries','previous_fifties','previous_zeros']].std()
            predict_bat = ((predict_bat-bat_means)/bat_std).tolist()
            bat_features.loc[:,['innings_played','previous_average','previous_strike_rate','previous_centuries','previous_fifties','previous_zeros']] = sc.fit_transform(bat_features.loc[:,['innings_played','previous_average','previous_strike_rate','previous_centuries','previous_fifties','previous_zeros']])

            predict_bat.insert(0,ven_bat[0])
            predict_bat.insert(0,opp_bat[0])

            #Array
            bat_features = bat_features.values
            bat_targets = bat_targets.values
            predict_bat_features = np.array(predict_bat).reshape(-1,1)
            predict_bat_features = predict_bat_features.T
            predict_bat_features = np.nan_to_num(predict_bat_features)

            print('\nBatting Parameters Tuning begins...')

            # Initializing Models
            #XGBoost
            if classes_bat > 2:
                bat_xgb = XGBClassifier(objective='multi:softmax',verbosity=0,silent=True)
            else:
                bat_xgb = XGBClassifier(objective='binary:logistic',verbosity=0, silent=True)
            bat_parameters_xgb = {'n_estimators':[75,100,125],'learning_rate':[0.1,0.01],'booster':['gbtree','dart']}
            #RandomForestClassifier
            if classes_bat > 2:
                bat_rfc = RandomForestClassifier(random_state=42)
                bat_parameters_rfc = {'n_estimators':[75,100,125],'criterion':['gini','entropy'],'min_samples_leaf':[1,2,3]}
            else:
                bat_rfc = RandomForestClassifier(random_state=42,min_samples_leaf=1)
                bat_parameters_rfc = {'n_estimators':[75,100,125],'criterion':['gini','entropy']}
            #SupportVectorMachine
            bat_svc = SVC()
            bat_parameters_svc = {'C':[1,5,10],'kernel':['rbf','linear','sigmoid'],'gamma':['auto','scale']}

            #ParameterTuningformodels
            bat_best_score, bat_best_params = None, None
                #XGBoost
            bat_gridsearch_xgb = GridSearchCV(estimator=bat_xgb,param_grid=bat_parameters_xgb,scoring='accuracy',cv=2)
            bat_gridresult_xgb = bat_gridsearch_xgb.fit(bat_features,bat_targets)
            bat_score, bat_params = gridsearchcv_results(bat_gridresult_xgb.cv_results_)
            bat_best_score, bat_best_params = [bat_score,'xgb'],bat_params
                #RandomForestClassifier
            bat_gridsearch_rfc = GridSearchCV(estimator=bat_rfc,param_grid=bat_parameters_rfc,scoring='accuracy',cv=2)
            bat_gridresult_rfc = bat_gridsearch_rfc.fit(bat_features,bat_targets)
            bat_score, bat_params = gridsearchcv_results(bat_gridresult_rfc.cv_results_)
            if bat_score > bat_best_score[0]:
                bat_best_score, bat_best_params = [bat_score,'rfc'],bat_params
                #SupportVectorMachine
            bat_gridsearch_svc = GridSearchCV(estimator=bat_svc,param_grid=bat_parameters_svc,scoring='accuracy',cv=2)
            bat_gridresult_svc = bat_gridsearch_svc.fit(bat_features,bat_targets)
            bat_score, bat_params = gridsearchcv_results(bat_gridresult_svc.cv_results_)
            if bat_score > bat_best_score[0]:
                bat_best_score, bat_best_params = [bat_score, 'svc'], bat_params

            print(f'Batting Prediction accuracy={bat_best_score[0]} with classifier={bat_best_score[1].upper()}')

            print('Batting Prediction begins...')

            #FinalModeling
                #XGBoost
            if bat_best_score[1] == 'xgb':
                if classes_bat > 2:
                    bat_classifier = XGBClassifier(objective='multi:softmax',n_estimators=bat_best_params['n_estimators'],learning_rate=bat_best_params['learning_rate'],booster=bat_best_params['booster'],verbosity=0,silent=True)
                else:
                    bat_classifier = XGBClassifier(objective='binary:logistic',min_leaf_samples=1,n_estimators=bat_best_params['n_estimators'], learning_rate=bat_best_params['learning_rate'], booster=bat_best_params['booster'], verbosity=0, silent=True)
                bat_classifier = bat_classifier.fit(bat_features,bat_targets)
                res['bat_prediction'] = bat_classifier.predict(predict_bat_features)
                #RandomForestClassifier
            elif bat_best_score[1] == 'rfc':
                if classes_bat > 2:
                    bat_classifier = RandomForestClassifier(n_estimators=bat_best_params['n_estimators'],criterion=bat_best_params['criterion'],random_state=42,min_samples_leaf=bat_best_params['min_samples_leaf'])
                else:
                    bat_classifier = RandomForestClassifier(n_estimators=bat_best_params['n_estimators'],criterion=bat_best_params['criterion'],random_state=42,min_samples_leaf=1)
                bat_classifier = bat_classifier.fit(bat_features,bat_targets)
                res['bat_prediction'] = bat_classifier.predict(predict_bat_features)
                #SupportVectorMachine
            elif bat_best_score[1] == 'svc':
                bat_classifier = SVC(C=bat_best_params['C'],kernel=bat_best_params['kernel'],gamma=bat_best_params['gamma'])
                bat_classifier = bat_classifier.fit(bat_features,bat_targets)
                res['bat_prediction'] = bat_classifier.predict(predict_bat_features)

            bat_runs = {'0':'0-10','1':'11-30','2':'31-50','3':'51-80','4':'81-120','5':'121-250'}
            res['bat_prediction'] = bat_runs[res['bat_prediction'][0]]

            print('Batting Prediction Ends!')

        else:
            print('NO Batting Prediction')

    else:
        print('No Batting Prediction')

    #BowlerPrediciton 
    if (param == 2 or param == 3):
        
        #Categorizing Runs
        bins = [0,1,3,5,7,10,11]
        labels = ['0','1','2','3','4','5']
        bowl_targets = pd.cut(bowl_targets['wickets'],bins,right=False,labels=labels,include_lowest=True)

        #Classification classes
        classes_bowl = len(bowl_targets.unique())

        if classes_bowl >= 2:
        
            #Categorizing Opposition and Venue
            le.fit(bowl_features['opposition'])
            opp_bowl = le.transform([opposition])
            bowl_features['opposition'] = le.transform(bowl_features['opposition'])
            le.fit(bowl_features['venue'])
            ven_bowl = le.transform([venue])
            bowl_features['venue'] = le.transform(bowl_features['venue'])

            predict_bowl = bowl_overall_details[['innings','average','strike_rate','economy','wicket_hauls']].values[0]

            #Scaling Non-Categorical Features
            bowl_means = bowl_features.loc[:,['innings_played','previous_average', 'previous_strike_rate', 'previous_economy','previous_wicket_hauls']].mean()
            bowl_std = bowl_features.loc[:,['innings_played','previous_average', 'previous_strike_rate', 'previous_economy','previous_wicket_hauls']].std()
            predict_bowl = ((predict_bowl-bowl_means)/bowl_std).tolist()
            bowl_features.loc[:,['innings_played','previous_average', 'previous_strike_rate', 'previous_economy','previous_wicket_hauls']] = sc.fit_transform(bowl_features.loc[:,['innings_played','previous_average', 'previous_strike_rate', 'previous_economy','previous_wicket_hauls']])

            predict_bowl.insert(0, ven_bowl[0])
            predict_bowl.insert(0, opp_bowl[0])

            #Array 
            bowl_features = bowl_features.values
            bowl_targets = bowl_targets.values
            predict_bowl_features = np.array(predict_bowl).reshape(-1,1)
            predict_bowl_features = predict_bowl_features.T
            predict_bowl_features = np.nan_to_num(predict_bowl_features)

            print('\nBowling Parameter Tuning begins...')

            # Initializing Models
                #XGBoost
            if classes_bowl  > 2:
                bowl_xgb = XGBClassifier(objective='multi:softmax',verbosity=0,silent=True)
            else:
                bowl_xgb = XGBClassifier(objective='binary:logistic',min_leaf_samples=1,verbosity=0,silent=True)
            bowl_parameters_xgb = {'n_estimators':[75,100,125],'learning_rate':[0.1,0.01],'booster':['gbtree','dart']}
                #RandomForestClassifier
            if classes_bowl > 2:
                bowl_rfc = RandomForestClassifier(random_state=42)
                bowl_parameters_rfc = {'n_estimators':[75,100,125],'criterion':['gini','entropy'],'min_samples_leaf':[1,2,3]}
            else:
                bowl_rfc = RandomForestClassifier(random_state=42,min_samples_leaf=1)
                bowl_parameters_rfc = {'n_estimators':[75,100,125],'criterion':['gini','entropy']}
                #SupportVectorMachine
            bowl_svc = SVC()
            bowl_parameters_svc = {'C':[1,5,10],'kernel':['rbf','linear','sigmoid'],'gamma':['auto','scale']}

            #ParameterTuningformodels
            bowl_best_score, bowl_best_params = None, None
                #XGBoost
            bowl_gridsearch_xgb = GridSearchCV(estimator=bowl_xgb,param_grid=bowl_parameters_xgb,scoring='accuracy',cv=2)
            bowl_gridresult_xgb = bowl_gridsearch_xgb.fit(bowl_features,bowl_targets)
            bowl_score, bowl_params = gridsearchcv_results(bowl_gridresult_xgb.cv_results_)
            bowl_best_score, bowl_best_params = [bowl_score, 'xgb'], bowl_params
                #RandomForestClassifier
            bowl_gridsearch_rfc = GridSearchCV(estimator=bowl_rfc,param_grid=bowl_parameters_rfc,scoring='accuracy',cv=2)
            bowl_gridresult_rfc = bowl_gridsearch_rfc.fit(bowl_features,bowl_targets)
            bowl_score, bowl_params = gridsearchcv_results(bowl_gridresult_rfc.cv_results_)
            if bowl_score > bowl_best_score[0]:
                bowl_best_score, bowl_best_params = [bowl_score, 'rfc'], bowl_params
                #SupportVectorMachine
            bowl_gridsearch_svc = GridSearchCV(estimator=bowl_svc,param_grid=bowl_parameters_svc,scoring='accuracy',cv=2)
            bowl_gridresult_svc = bowl_gridsearch_svc.fit(bowl_features,bowl_targets)
            bowl_score, bowl_params = gridsearchcv_results(bowl_gridresult_rfc.cv_results_)
            if bowl_score > bowl_best_score[0]:
                bowl_best_score, bowl_best_params = [bowl_score, 'svc'], bowl_params

            print(f'The bowling prediction accuracy={bowl_best_score[0]} with classifier={bowl_best_score[1].upper()}')

            print('Bowling Prediction begins...')

            #FinalModeling
                #XGBoost
            if bowl_best_score[1] == 'xgb':
                if classes_bowl > 2:
                    classifier = XGBClassifier(objective='multi:softmax',n_estimators=bowl_best_params['n_estimators'],learning_rate=bowl_best_params['learning_rate'],booster=bowl_best_params['booster'],verbosity=0,silent=True)
                else:
                    classifier = XGBClassifier(objective='binary:logistic',min_leaf_samples=1,n_estimators=bowl_best_params['n_estimators'],learning_rate=bowl_best_params['learning_rate'],booster=bowl_best_params['booster'],verbosity=0,silent=True)
                classifier = classifier.fit(bowl_features,bowl_targets)
                res['bowl_prediction'] = classifier.predict(predict_bowl_features)
                #RandomForestClassifier
            elif bowl_best_score[1] == 'rfc':
                if classes_bowl > 2:
                    classifier = RandomForestClassifier(n_estimators=bowl_best_params['n_estiamtors'],criterion=bowl_best_params['criterion'],random_state=42,min_samples_leaf=bowl_best_params['min_leaf_samples'])
                else:
                    classifier = RandomForestClassifier(n_estimators=bowl_best_params['n_estimators'],criterion=bowl_best_params['criterion'],random_state=42,min_samples_leaf=1)
                classifier = classifier.fit(bowl_features,bowl_targets)
                res['bowl_prediction'] = classifier.predict(predict_bowl_features)
                #SupportVectorMachine
            elif bowl_best_score[1] == 'svc':
                classifier = SVC(C=bowl_best_params['C'],kernel=bowl_best_params['kernel'],gamma=bowl_best_params['gamma'])
                classifier = classifier.fit(bowl_features,bowl_targets)
                res['bowl_prediction'] = classifier.predict(predict_bowl_features)

            bowl_wickets = {'0':'0','1':'1-2','2':'3-4','3':'5-6','4':'7-9','5':'10'}
            res['bowl_prediction'] = bowl_wickets[res['bowl_prediction'][0]]

            print('Bowling Prediction Ends!')

        else:
            print('NO bwoling prediction')
    
    else:
        print('No bowling prediction')

    return res
