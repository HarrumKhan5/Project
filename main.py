# Import necessary libraries
import pandas as pd #data manipulation library
import numpy as np #data manipulation library
import matplotlib.pyplot as plt #data visualization library
import seaborn as sns #data visualization library
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("Dataset.csv")
print(data.head())
print(data.shape)
print("")
print(data.columns)
print("")
print(data.dtypes)
print("")
print(data.info())
print(data.Weather.value_counts())
print(data.Weather.unique())
print(data.Weather.nunique())

###Convertig the Weather categories into Standard categories
x="Thunderstorms,Modern Rain Showers,Fog"
list_of_lists=[w.split()for w in x.split(",")]
print(list_of_lists)
from itertools import chain
flat_list=list(chain(*list_of_lists))
print(flat_list)
def Create_list(x):
    list_of_lists=[w.split()for w in x.split(",")]
    flat_list=list(chain(*list_of_lists))
    return flat_list
def Get_Weather(list1):
    if "Fog" in list1 and "Rain" in list1:
        return "RAIN+FOG"
    elif "Snow" in list1 and "Rain" in list1:
        return"SNOW+FOG"
    elif "Snow" in list1:
        return "Snow"
    elif "Rain" in list1:
        return "RAIN"
    elif "Fog" in list1:
        return "FOG"
    elif "Clear" in list1:
        return "Clear"
    elif "Cloudy" in list1:
        return "Cloudy"
    else:
        return "RAIN"
Create_list(x)
Get_Weather(Create_list(x))
data["Std_Weather"]=data["Weather"].apply(lambda x: Get_Weather(Create_list(x)))
print(data.head())
print(data.Std_Weather.value_counts())

##Samplen Selection adn data balancing:
cloudy_df = data[data['Std_Weather'] == 'Cloudy']
print(cloudy_df)
cloudy_df_sample = cloudy_df.sample(600)
print(cloudy_df_sample.shape)
clear_df = data[data['Std_Weather'] == 'Clear'].sample(600)
print(clear_df.shape)

##Dataset Balancing
rain_df = data[data['Std_Weather'] == 'RAIN']
snow_df = data[data['Std_Weather'] == 'SNOW']
print(rain_df.shape)
print(snow_df.shape)

##Create New Weather Dataset
weather_df = pd.concat([cloudy_df_sample, clear_df, rain_df, snow_df], axis=0)
print(weather_df.head())
print(weather_df.shape)
print(weather_df.Std_Weather.value_counts())
weather_df.drop(columns=['Date/Time', 'Weather'], axis=1, inplace=True)
print(weather_df.head())
print(weather_df[weather_df.duplicated()])
weather_df.isnull().sum()
print(weather_df.dtypes)
print(weather_df.describe())

##Correlation among the features:
cols = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h',
'Visibility_km', 'Press_kPa']
cor_matrix = weather_df[cols].corr()
print(cor_matrix)

##Heat map 
sns.heatmap(cor_matrix, annot = True)
data_hist_plot = weather_df.hist(figsize = (20,20), color = "#5F9EA0")
num_cols = weather_df.select_dtypes(exclude = ['object']).columns.tolist()
print(num_cols)
fig, axes = plt.subplots(ncols = 6, figsize = (12,3))
for column, axis in zip(num_cols, axes):
    sns.boxplot(data = weather_df[column], ax = axis)
    axis.set_title(column)
plt.tight_layout()
plt.show()

from sklearn.preprocessing import LabelEncoder  
label_Encoder = LabelEncoder()  
weather_df['Std_Weather'] = label_Encoder.fit_transform(weather_df['Std_Weather'])
print(label_Encoder.classes_)  
cat_code=dict(zip(label_Encoder.classes_,label_Encoder.transform(label_Encoder.classes_)))
print(cat_code)  
print(weather_df.Std_Weather.value_counts()) 

##x,y variables:  
###Indepen dent variable  
x = weather_df.drop(['Std_Weather'], axis = 1)

###Target Variable
y=weather_df["Std_Weather"]
print(weather_df.head())

###Feature Scalling:
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(x)
print(X_std)

##Splitting Data into training and testing:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape)

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

### 1. Balance the Dataset using SMOTE
X, y = weather_df.drop("Std_Weather", axis=1), weather_df["Std_Weather"]
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

##2. Feature Selection
feature_selector = ExtraTreesClassifier(random_state=42)
feature_selector.fit(X_balanced, y_balanced)
important_features = X.columns[feature_selector.feature_importances_ > 0.1]
X_balanced = X_balanced[important_features]

##3. Train-Test Split with Stratification
x_train, x_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
)

##4. Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

##5. Hyperparameter Tuning for Random Forest
rf_param_grid = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [None, 10, 20, 30],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5, 10],            # Added min samples split
    "min_samples_leaf": [1, 2, 4], 
    "criterion": ["gini", "entropy"],
}
rf_random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=rf_param_grid,
    n_iter=100,
    scoring="accuracy",
    cv=5,
    random_state=42,
    n_jobs=-1,
)
rf_random_search.fit(x_train, y_train)

##Best Random Forest Model
best_rf = rf_random_search.best_estimator_

##6. Model Ensemble (VotingClassifier)
from xgboost import XGBClassifier
xgb_clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="mlogloss")
voting_clf = VotingClassifier(
    estimators=[
        ("rf", best_rf),
        ("gbc", GradientBoostingClassifier(random_state=42)),
        ("abc", AdaBoostClassifier(random_state=42)),
    ],
    voting="soft",
)
voting_clf.fit(x_train, y_train)

##7. Evaluate the Model
y_pred = voting_clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Voting Classifier: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

##8. Weather Prediction System
Temp = float(input("Enter the Temp_C: "))
dpt = float(input("Enter the Dew Point Temp_C: "))
rh = float(input("Enter the Relative Humidity (%): "))
ws = float(input("Enter the Wind Speed (km/h): "))
vs = float(input("Enter the Visibility (km): "))
pr = float(input("Enter the Pressure (kPa): "))

##Scaling
input_data = [Temp, dpt, rh, ws, vs, pr]
scaled_data = scaler.transform([input_data])
prediction = voting_clf.predict(scaled_data)

##Output Prediction
categories = {0: "CLEAR", 1: "CLOUDY", 2: "RAINY", 3: "SNOWY"}
print(f"Predicted Weather: {categories[prediction[0]]}")
print(accuracy_score(y_test, y_pred))

##Cross validation
from sklearn.model_selection import StratifiedKFold, cross_val_score
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(voting_clf, X_balanced, y_balanced, cv=skf, scoring="accuracy")
print(f"Cross-Validated Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")