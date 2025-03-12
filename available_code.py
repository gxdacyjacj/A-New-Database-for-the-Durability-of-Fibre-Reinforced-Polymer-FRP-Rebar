#%%
# import
import numpy as np
import pandas as pd
import re
import numpy
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, datasets
import statsmodels.api as sm
import math
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from collections import Counter
import math

# %% data input
data_ori = pd.read_excel('Database.xlsx',skiprows=2,index_col=6,header=[0,1])
data_ori = data_ori.iloc[:,1:]
data_ori.head
data_ori.iloc[148,10]
# %%
# preprocessing
#change SMD to NAN
def ChangeSMDToNAN(d):
    length, width = d.shape
    dnew = pd.DataFrame(index=d.index.copy(), columns=d.columns.copy())
    for j in range(width):
        for i in range(length):
            if d.iloc[i,j] == 'SMD':
                dnew.iloc[i,j] = np.nan
            elif d.iloc[i,j] == 'Notreported':
                dnew.iloc[i,j] = 'Unknown'
            else:
                dnew.iloc[i,j] = d.iloc[i,j]
    return dnew
data_ori = ChangeSMDToNAN(data_ori)
# %% change range to mean
length = len(data_ori.iloc[:,9])
for j in [('Rebar_information','glass_transition_temperature'),
          ('Rebar_information','glass_transition_temperature_run_2'),
          ('Rebar_information','cure_ratio'),
          ('Rebar_information','Fiber_content_weight'),
          ('Rebar_information','Fiber_content_volume'),
          ('Rebar_information','Void_content'),
          ('Rebar_information','diameter'),
          ('Rebar_information','average_area'),
          ('Rebar_information','nominal_area'),
          ('Mechanical_property_of_control_groups','num'),
          ('Condition_environment','temperature'),
          ('Condition_environment','pH_of_concrete'),
          ('Condition_environment','strength_of_concrete'),
          ('Condition_environment','crack'),
          ('solution_condition','pH'),
          ('solution_condition','pHafter'),
          ('moisture_condition','RH'),
          ('field_condition','field_average_humidity'),
          ('field_condition','field_average_temperature'),
          ('Cycle_condition','temp'),
          ('Cycle_condition','temp2'),
          ('Load','value'),
          ('Mechanical_result','num')]:
    for i in range(1,length+1):
        if  type(data_ori.loc[i,j]) == str:
            a = ',' in data_ori.loc[i,j] 
            b = ':' not in data_ori.loc[i,j]
            if a and b:
                newvalue = np.mean(list(map(float,re.findall(r"\d+\.?\d*",data_ori.loc[i,j]))))
                if ~np.isnan(newvalue):
                    data_ori.loc[i,j] = newvalue
data_ori.iloc[148,10]
#%% 
# some basic assumption
data_ori['selected_feature','pH_of_condition_enviroment'] = np.nan
data_ori['selected_feature','Chloride_ion'] = np.nan
data_ori.loc[data_ori.loc[:,('solution_condition','pH')] == 'tap water',('selected_feature','Chloride_ion')] = 0
data_ori.loc[data_ori.loc[:,('solution_condition','pH')] == 'sea water',('selected_feature','Chloride_ion')] = 1
data_ori.loc[data_ori.loc[:,('solution_condition','pH')] == 'distilled water',('selected_feature','Chloride_ion')] = 0
data_ori.loc[data_ori.loc[:,('solution_condition','pH')] == 'deionized water',('selected_feature','Chloride_ion')] = 0
data_ori.loc[data_ori.loc[:,('solution_condition','pH')] == 'DI water',('selected_feature','Chloride_ion')] = 0
data_ori.loc[data_ori.loc[:,('solution_condition','pH')] == 'tap water',('selected_feature','pH_of_condition_enviroment')] = 7
data_ori.loc[data_ori.loc[:,('solution_condition','pH')] == 'sea water',('selected_feature','pH_of_condition_enviroment')] = 7
data_ori.loc[data_ori.loc[:,('solution_condition','pH')] == 'distilled water',('selected_feature','pH_of_condition_enviroment')] = 7
data_ori.loc[data_ori.loc[:,('solution_condition','pH')] == 'deionized water',('selected_feature','pH_of_condition_enviroment')] = 7
data_ori.loc[data_ori.loc[:,('solution_condition','pH')] == 'DI water',('selected_feature','pH_of_condition_enviroment')] = 7
#%% encoding
data_ori['selected_feature','concrete'] = np.nan
data_ori['selected_feature','diameter'] = np.nan
data_ori['selected_feature','load_value'] = np.nan
data_ori['selected_feature','fiber_content'] = np.nan
# concrete = 1
for i in range(1,length+1):
    if (type(data_ori.loc[i,('Condition_environment','concrete')]) == str) or \
    (type(data_ori.loc[i,('Condition_environment','crack')]) == str) or \
    (type(data_ori.loc[i,('Condition_environment','cover')]) == str):
        data_ori.loc[i,('selected_feature','concrete')] = 1
    elif (~np.isnan(data_ori.loc[i,('Condition_environment','crack')])) or \
    (~np.isnan(data_ori.loc[i,('Condition_environment','cover')])):
        data_ori.loc[i,('selected_feature','concrete')] = 1
    else:
        data_ori.loc[i,('selected_feature','concrete')] = 0
# diameter
    if data_ori.loc[i,('Rebar_information','nominal_area')] != 'Unknown':
        data_ori.loc[i,('selected_feature','diameter')] = 2 * (data_ori.loc[i,('Rebar_information','nominal_area')]/numpy.pi)**0.5
    if type(data_ori.loc[i,('Rebar_information','diameter')]) == int or type(data_ori.loc[i,('Rebar_information','diameter')]) == float:
        data_ori.loc[i,('selected_feature','diameter')] = data_ori.loc[i,('Rebar_information','diameter')]
# pH
    if type(data_ori.loc[i,('solution_condition','pH')]) == int or type(data_ori.loc[i,('solution_condition','pH')]) == float or type(data_ori.loc[i,('solution_condition','pH')]) == np.float64:
        data_ori.loc[i,('selected_feature','pH_of_condition_enviroment')] = data_ori.loc[i,('solution_condition','pH')]
#ph after
    if data_ori.loc[i,('solution_condition','pHafter')] != 'Unknown' and data_ori.loc[i,('solution_condition','pHafter')] != 'sea water':
        if ~pd.isna(data_ori.loc[i,('solution_condition','pHafter')]) :
            data_ori.loc[i,('selected_feature','pH_of_condition_enviroment')] = (data_ori.loc[i,('selected_feature','pH_of_condition_enviroment')]+data_ori.loc[i,('solution_condition','pHafter')])/2    
# stress/strain
    if type(data_ori.loc[i,('Load','note')]) == float and data_ori.loc[i,('Load','note')] < 1:
        data_ori.loc[i,('selected_feature','load_value')] = data_ori.loc[i,('Load','note')]
    elif data_ori.loc[i,('Load','stress_or_strain')] == 'stress': 
        if type(data_ori.loc[i,('Load','value')]) == str:
            data_ori.loc[i,('selected_feature','load_value')] = float(re.findall(r"\d+\.?\d*",data_ori.loc[i,('Load','value')])[0])
        else:
            data_ori.loc[i,('selected_feature','load_value')] = data_ori.loc[i,('Load','value')]/data_ori.loc[i,('Load','ultimate_tensile_strength')]
    elif data_ori.loc[i,('Load','stress_or_strain')] == 'strain':
        data_ori.loc[i,('selected_feature','load_value')] = data_ori.loc[i,('Load','value')]*0.001*data_ori.loc[i,('Load','tensile_modulus')]/data_ori.loc[i,('Load','ultimate_tensile_strength')]
    else:
        data_ori.loc[i,('selected_feature','load_value')] = 0    
# ingredient
    if type(data_ori.loc[i,('solution_condition','ingredient')]) == str:
        if 'Cl' in data_ori.loc[i,('solution_condition','ingredient')]:
            data_ori.loc[i,('selected_feature','Chloride_ion')] = 1
    if type(data_ori.loc[i,('Condition_environment','concrete')]) == str:
        if 'seawater' in data_ori.loc[i,('Condition_environment','concrete')]:
            data_ori.loc[i,('selected_feature','Chloride_ion')] = 1                                     
    if  math.isnan(data_ori.loc[i,('selected_feature','Chloride_ion')]):
        data_ori.loc[i,('selected_feature','Chloride_ion')] = 0
# preloading = 0
    if data_ori.loc[i,('Load','type_of_load')] == 'preloading':
        data_ori.loc[i,('selected_feature','load_value')] = 0
# pH of concrete to pH of condition environment
    if data_ori.loc[i,('selected_feature','concrete')] == 1:
        if data_ori.loc[i,('Condition_environment','pH_of_concrete')] == 'Unknown':
            data_ori.loc[i,('selected_feature','pH_of_condition_enviroment')] = 13
        else:
            data_ori.loc[i,('selected_feature','pH_of_condition_enviroment')] = data_ori.loc[i,('Condition_environment','pH_of_concrete')]

#    change volume content to weight content
    if data_ori.loc[i,('Rebar_information','Fiber_content_weight')] == 'Unknown':
        if type(data_ori.loc[i,('Rebar_information','Fiber_content_volume')]) == int or type(data_ori.loc[i,('Rebar_information','Fiber_content_volume')]) == float or type(data_ori.loc[i,('Rebar_information','Fiber_content_volume')]) == np.float64:
            if data_ori.loc[i,('Rebar_information','Fiber_type')] == 'Glass':
                density_fiber = 2.55
            elif data_ori.loc[i,('Rebar_information','Fiber_type')] == 'Carbon':
                density_fiber = 1.84
            elif data_ori.loc[i,('Rebar_information','Fiber_type')] == 'Basalt':
                density_fiber = 2.67
            else:
                density_fiber = 2.0
            if data_ori.loc[i,('Rebar_information','Matrix_type')] == 'Vinyl ester':
                density_matrix = 1.09
            elif data_ori.loc[i,('Rebar_information','Matrix_type')] == 'Epoxy':
                density_matrix = 1.1
            elif data_ori.loc[i,('Rebar_information','Matrix_type')] == 'Polyester':
                density_matrix = 1.38
            else:
                density_matrix = 1.2
            
            data_ori.loc[i,('selected_feature','fiber_content')] = (100.0*data_ori.loc[i,('Rebar_information','Fiber_content_volume')]*density_fiber)/(data_ori.loc[i,('Rebar_information','Fiber_content_volume')]*density_fiber+(100.0-data_ori.loc[i,('Rebar_information','Fiber_content_volume')])*density_matrix)
    elif type(data_ori.loc[i,('Rebar_information','Fiber_content_weight')]) == int or float or np.float64:
        data_ori.loc[i,('selected_feature','fiber_content')] = data_ori.loc[i,('Rebar_information','Fiber_content_weight')]
# only select solution condition        
    if pd.isna(data_ori.loc[i,('solution_condition','pH')]):
        data_ori.loc[i,('selected_feature','pH_of_condition_enviroment')] = np.nan
# surface_treatment
    if data_ori.loc[i,('Rebar_information', 'surface_treatment')] == 'sand coated':
        data_ori.loc[i,('selected_feature','surface_treatment')] = 0
    elif data_ori.loc[i,('Rebar_information', 'surface_treatment')] == 'Smooth':
        data_ori.loc[i,('selected_feature','surface_treatment')] = 1
#  max strength
    if type(data_ori.loc[i,('Mechanical_property_of_control_groups', 'Value1')]) == (int or float or np.float64):
        data_ori.loc[i,('selected_feature','max_strength')] = data_ori.loc[i,('Mechanical_property_of_control_groups', 'Value1')]
#  glass transition temperature
    if type(data_ori.loc[i,('Rebar_information', 'glass_transition_temperature')]) == (int or float or np.float64):
        data_ori.loc[i,('selected_feature','glass_transition_temperature')] = data_ori.loc[i,('Rebar_information', 'glass_transition_temperature')]

data_ori['selected_feature','Glass_or_Basalt'] = np.nan
data_ori['selected_feature','Vinyl_ester_or_Epoxy'] = np.nan
for i in range(1,length+1):
    if data_ori.loc[i,('Rebar_information','Fiber_type')] == 'Glass':
        data_ori.loc[i,('selected_feature','Glass_or_Basalt')] = 1
    elif data_ori.loc[i,('Rebar_information','Fiber_type')] == 'Basalt':
        data_ori.loc[i,('selected_feature','Glass_or_Basalt')] = 0
    if data_ori.loc[i,('Rebar_information','Matrix_type')] == 'Vinyl ester':
        data_ori.loc[i,('selected_feature','Vinyl_ester_or_Epoxy')] = 1
    elif data_ori.loc[i,('Rebar_information','Matrix_type')] == 'Epoxy':
        data_ori.loc[i,('selected_feature','Vinyl_ester_or_Epoxy')] = 0

data_ori['selected_feature','condition_time'] = data_ori['Condition_environment','time']
data_ori['selected_feature','Temperature'] = data_ori['Condition_environment','temperature']
data_ori['selected_feature','Tensile_strength_retention'] = data_ori['Mechanical_result','retention1']
data_ori['selected_feature','Target_parameter'] = data_ori['Target_parameter','Target_parameter']



# %%
i0_data_pick = pd.DataFrame(
    {
        'Title' : data_ori['Source_information', 'Title'].tolist(),
        'Target parameter' : data_ori['selected_feature','Target_parameter'].tolist(),
        'Tensile strength retention' : data_ori['selected_feature','Tensile_strength_retention'].tolist(),
        'pH of condition environment' : data_ori['selected_feature','pH_of_condition_enviroment'].tolist(),
        'Exposure time' : data_ori['selected_feature','condition_time'].tolist(),
        'Fiber content' : data_ori['selected_feature','fiber_content'].tolist(),
        'Exposure temperature' : data_ori['selected_feature','Temperature'].tolist(),
        'Diameter' : data_ori['selected_feature','diameter'].tolist(),
        'Chloride ion' : data_ori['selected_feature','Chloride_ion'].tolist(),
        'Concrete' : data_ori['selected_feature','concrete'].tolist(),
        'Load' : data_ori['selected_feature','load_value'].tolist(),
        'Glass or Basalt' : data_ori['selected_feature','Glass_or_Basalt'].tolist(),
        'Vinyl ester or Epoxy' : data_ori['selected_feature','Vinyl_ester_or_Epoxy'].tolist(),
        'Sand coated or Smooth' : data_ori['selected_feature','surface_treatment'].tolist(),
        'Maximum strength' : data_ori['selected_feature','max_strength'].tolist(),
        'Glass transition temperature' : data_ori['selected_feature','glass_transition_temperature'].tolist()
    }
    ,index=data_ori.index)
# %% 
i01_data_pick = i0_data_pick[i0_data_pick.loc[:,'Target parameter'] == 'Tensile']
i010_data_pick = i01_data_pick.loc(axis=1)['Title'
                                           ,'Tensile strength retention'
                                           ,'pH of condition environment'
                                           ,'Exposure time'
                                           ,'Fiber content'
                                           ,'Exposure temperature'
                                           ,'Diameter'
                                           ,'Concrete'
                                           ,'Load'
                                           ,'Chloride ion'
                                           ,'Glass or Basalt'
                                           ,'Vinyl ester or Epoxy'
                                           ,'Sand coated or Smooth'
                                           ,'Maximum strength'
 #                                          ,'Glass transition temperature'
                                         ]
i010_data_pick = i010_data_pick.dropna()
#%%
'''
Use permutation importance to measure the importance of each feature
'''
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
#%%
rng = np.random.RandomState(0)
#%%
#change colume name
i010_data_pick.columns = ['Title','Tensile strength retention', 'pH of condition environment',
       'Exposure time', 'Fibre content', 'Exposure temperature', 'Diameter',
       'Presence of concrete', 'Load', 'Presence of chloride ion', 'Fibre type',
       'Matrix type', 'Surface treatment', 'Strength of unconditioned rebar']
#%%
X = i010_data_pick[i010_data_pick.columns.drop(['Title', 'Tensile strength retention'])].astype(float)
y = i010_data_pick['Tensile strength retention'].astype(float)
#%%
i010_data_pick.columns.drop(['Title', 'Tensile strength retention'])
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

kf = KFold(n_splits=5, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X_train):
    xgb_model = xgb.XGBRegressor(n_jobs=1).fit(X_train.iloc[train_index], y_train.iloc[train_index])
    preditions = xgb_model.predict(X_train.iloc[test_index])
    acturals = y_train.iloc[test_index]
    print(mean_squared_error(acturals, preditions))
    print(r2_score(acturals, preditions))


print('parameter optimization')
rf_model = RandomForestRegressor(random_state=rng)
rf_clf = GridSearchCV(
    rf_model,
    {"max_depth": [2, 4, 8], "n_estimators": [10, 50, 100]},
    verbose=1,
    n_jobs=1,
    cv=5,
)
rf_clf.fit(X_train, y_train)
print(rf_clf.best_score_)
print(rf_clf.best_params_)
rf_model_po = RandomForestRegressor(**rf_clf.best_params_,random_state=rng).fit(X_train, y_train)


print('parameter optimization')
xgb_model = xgb.XGBRegressor(random_state=rng)
xgb_clf = GridSearchCV(
    xgb_model,
    {"max_depth": [2, 4, 8], "n_estimators": [10, 50, 100]},
    verbose=1,
    n_jobs=1,
    cv=5,
)
xgb_clf.fit(X_train, y_train)
print(xgb_clf.best_score_)
print(xgb_clf.best_params_)
xgb_model_po = RandomForestRegressor(**xgb_clf.best_params_,random_state=rng).fit(X_train, y_train)


print('parameter optimization')
lgb_model = lgb.LGBMRegressor(force_row_wise=True, random_state=rng)
lgb_clf = GridSearchCV(
    lgb_model,
    {"max_depth": [2, 4, 8], "n_estimators": [10, 50, 100]},
    verbose=1,
    n_jobs=1,
    cv=5,
)
lgb_clf.fit(X_train, y_train)
print(lgb_clf.best_score_)
print(lgb_clf.best_params_)
lgb_model_po = RandomForestRegressor(**lgb_clf.best_params_,random_state=rng).fit(X_train, y_train)

lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

from sklearn.inspection import permutation_importance
rf_imp = permutation_importance(rf_model,X_test,y_test,n_repeats=30, random_state = rng)
for i in rf_imp.importances_mean.argsort()[::-1]:
    print(f"{X.columns[i]:<8}"
            f"{rf_imp.importances_mean[i]:.3f}"
            f" +/- {rf_imp.importances_std[i]:.3f}")

xgb_imp = permutation_importance(xgb_model,X_test,y_test,n_repeats=30, random_state = rng)
for i in xgb_imp.importances_mean.argsort()[::-1]:
    print(f"{X.columns[i]:<8}"
            f"{xgb_imp.importances_mean[i]:.3f}"
            f" +/- {xgb_imp.importances_std[i]:.3f}")

lgb_imp = permutation_importance(lgb_model,X_test,y_test,n_repeats=30, random_state = rng)
for i in lgb_imp.importances_mean.argsort()[::-1]:
    print(f"{X.columns[i]:<8}"
            f"{lgb_imp.importances_mean[i]:.3f}"
            f" +/- {lgb_imp.importances_std[i]:.3f}")

#%%
import matplotlib.pyplot as plt

def plot_permutation_importance(imp, X, ax, highlight_features=None):
    perm_sorted_idx = imp.importances_mean.argsort()

    box = ax.boxplot(
        imp.importances[perm_sorted_idx].T,
        vert=False,
        labels=X.columns[perm_sorted_idx],
        patch_artist=True,  # Enable color modifications
        # All median lines should be black
        medianprops=dict(color="black")
    )

    ax.axvline(x=0, color="k", linestyle="--")
    # Set all boxes to transparent
    for patch in box['boxes']:
        patch.set(facecolor="none")  # Makes the box transparent
        patch.set_edgecolor("black")  # Default black outline

    # If highlight_features is provided, color the specific boxes and median lines
    if highlight_features:
        feature_labels = X.columns[perm_sorted_idx]
        highlight_indices = [i for i, label in enumerate(feature_labels) if label in highlight_features]

        # Change color of the selected boxes and median lines
        for idx in highlight_indices:
            box['boxes'][idx].set_edgecolor('red')  # Change box outline to red
            box['medians'][idx].set_color('red')  # Change median line to red
            # Change the color of the labels to red
            ax.get_yticklabels()[idx].set_color('red')
            # change the color of the whiskers to red
            box['whiskers'][2*idx].set_color('red')
            box['whiskers'][2*idx+1].set_color('red')
            # change the color of the caps to red
            box['caps'][2*idx].set_color('red')
            box['caps'][2*idx+1].set_color('red')
            # change the color of outliers to red
            box['fliers'][idx].set(markeredgecolor='red', marker='o', markersize=5)

    
    return ax

# Features to highlight
highlight_features = ['Presence of concrete', 'Load', 'Presence of chloride ion', 
                      'Fibre type', 'Matrix type', 'Surface treatment', 
                      'Strength of unconditioned rebar']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")

# Random Forest
plot_permutation_importance(rf_imp, X, ax1, highlight_features)
ax1.set_xlabel('Decrease in accuracy score')
ax1.set_title('Random Forest', fontsize=16, x=0.73, y=0.05, pad=-14)

# XGBoost
plot_permutation_importance(xgb_imp, X, ax2, highlight_features)
ax2.set_xlabel('Decrease in accuracy score')
ax2.set_title('XGBoost', fontsize=16, x=0.83, y=0.05, pad=-14)

# LightGBM
plot_permutation_importance(lgb_imp, X, ax3, highlight_features)
ax3.set_xlabel('Decrease in accuracy score')
ax3.set_title('LightGBM', fontsize=16, x=0.82, y=0.05, pad=-14)

plt.show()

#%%
'''
compare the performance of three models based on different size dataset
'''
# %% small dataset using the same references with D1
data_small = pd.read_excel('database1.xlsx')
#%% selection base on the same research scope
i011_data_pick = i010_data_pick[(i010_data_pick['pH of condition environment'] > 9) & 
                      (i010_data_pick['Load'] == 0 ) &     
                      (i010_data_pick['Fibre type'] == 1)                                
                      ].dropna()
i011_data_pick = i011_data_pick.drop(columns=['Presence of concrete', 'Load', 'Presence of chloride ion', 'Fibre type',
       'Matrix type', 'Surface treatment', 'Strength of unconditioned rebar'])
#%%
title_list = list(dict(Counter(i011_data_pick['Title'])).keys())
title_list_small = list(dict(Counter(data_small['Title'])).keys())
title_different = [i for i in title_list if i not in title_list_small]
small = i011_data_pick[i011_data_pick['Title'].isin(title_list_small)]
different = i011_data_pick[i011_data_pick['Title'].isin(title_different)]
#%%
def Split_by_title(df,target_column,title_column,ratio,random_seed=np.nan):
    np.random.seed() if np.isnan(random_seed)  else np.random.seed(random_seed)
    title_list = list(dict(Counter(df[title_column])).keys())
    title_len = len(title_list)
    num_test = round(ratio*title_len)
    test_title = list(np.random.choice(title_list,num_test,replace=False))
    train_title = [i for i in title_list if i not in test_title]
    train = df[df[title_column].isin(train_title)]
    test = df[df[title_column].isin(test_title)]
    X_train = train.drop(columns=[title_column,target_column])
    X_test = test.drop(columns=[title_column,target_column])
    y_train = train.loc(axis=1)[target_column]
    y_test = test.loc(axis=1)[target_column]
    return X_train,X_test,y_train,y_test
#%%

repeats = 100
e = np.zeros((3,2,repeats))
for i in range(repeats):
    X_train_small,X_test_small,Y_train_small,Y_test_small = Split_by_title(small,'Tensile strength retention','Title',0.3)
    X_train_different,X_test_different,Y_train_different,Y_test_different = Split_by_title(different,'Tensile strength retention','Title',0.3)
    X_train_large= pd.concat([X_train_different,X_train_small])
    X_test_large= pd.concat([X_test_different,X_test_small])
    Y_train_large=pd.concat([Y_train_different,Y_train_small])
    Y_test_large = pd.concat([Y_test_different, Y_test_small])
    X_train_small = X_train_small.astype(float)
    Y_train_small = Y_train_small.astype(float)
    X_train_large = X_train_large.astype(float)
    Y_train_large = Y_train_large.astype(float)
    X_test_large = X_test_large.astype(float)
    Y_test_large = Y_test_large.astype(float)
    X_test_small = X_test_small.astype(float)
    Y_test_small = Y_test_small.astype(float)
    
    model = lgb.LGBMRegressor()
    model.fit(X_train_small, Y_train_small)
    e[2,0,i] = mean_absolute_error(Y_test_large,model.predict(X_test_large))
    model = lgb.LGBMRegressor()
    model.fit(X_train_large, Y_train_large)
    e[2,1,i] = mean_absolute_error(Y_test_large,model.predict(X_test_large))
    a = model.predict(X_test_large)


    model = xgb.XGBRegressor()
    model.fit(X_train_small, Y_train_small)
    e[1,0,i] = mean_absolute_error(Y_test_large,model.predict(X_test_large))
    model = xgb.XGBRegressor()
    model.fit(X_train_large, Y_train_large)
    e[1,1,i] = mean_absolute_error(Y_test_large,model.predict(X_test_large))
    b = model.predict(X_test_large)

    model = RandomForestRegressor()
    model.fit(X_train_small, Y_train_small)
    e[0,0,i] = mean_absolute_error(Y_test_large,model.predict(X_test_large))
    model = RandomForestRegressor()
    model.fit(X_train_large, Y_train_large)
    e[0,1,i] = mean_absolute_error(Y_test_large,model.predict(X_test_large))
    c = model.predict(X_test_large)

e_mean = np.mean(e,axis=2)
e_std = np.std(e,axis=2)

# %%
#%%
error = pd.DataFrame(e_mean, index=['Random Forest','XGBoost','LightGBM'],
                 columns=['Small Train','Large Train'])
# draw bar chart using error
# Create bar chart
x = np.arange(len(error.index))
width = 0.35

fig, ax = plt.subplots(figsize=(8,6))
rects1 = ax.bar(x - width/2, error['Small Train'], width, label='Small Training Dataset', color='orange')
rects2 = ax.bar(x + width/2, error['Large Train'], width, label='Large Training Dataset', color='blue')

# Customize chart
ax.set_ylim(0,0.16)
ax.set_ylabel('Mean Absolute Error')
#ax.set_title('Error Comparison by Model and Test Scenario')
ax.set_xticks(x)
ax.set_xticklabels(error.index)
ax.legend()

# Add value labels on top of bars
ax.bar_label(rects1, padding=3, fmt='%.3f')
ax.bar_label(rects2, padding=3, fmt='%.3f')

plt.tight_layout()
plt.show()

# %%
