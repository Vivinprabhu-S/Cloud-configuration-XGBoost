import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor 
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import mean_squared_error
from xgboost import plot_importance
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('./cloud.csv')

# Describe the dataset
print(df.describe())

df = df.rename(columns={
    'id': 'id',
    'timestamp': 'timestamp',
    'numberOfInstances': 'num_instances',
    'instanceName': 'instance_name',
    'vcpu': 'vcpu',
    'cpuUsage': 'cpu_usage',
    'memory': 'memory',
    'memoryUsage': 'memory_usage',
    'networkPerformance': 'network_performance',
    'storageType': 'storage_type',
    'storageSize': 'storage_size',
    'costPerHour': 'cost_per_hour',
    'target': 'target'
})

X = df.drop(columns=['cost_per_hour'])
y = df['cost_per_hour']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

categorical_cols = ['storage_type', 'instance_name']
numeric_cols = ['num_instances', 'vcpu', 'cpu_usage', 'memory', 'memory_usage', 'storage_size']

# Preprocess
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', TargetEncoder(), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

# Pipeline
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', XGBRegressor(random_state=8)) 
])

# Bayesian optimization
search_space = {
    'clf__max_depth': Integer(2, 6),
    'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__colsample_bylevel': Real(0.5, 1.0),
    'clf__colsample_bynode': Real(0.5, 1.0),
    'clf__reg_alpha': Real(0.0, 10.0),
    'clf__reg_lambda': Real(0.0, 10.0),
    'clf__gamma': Real(0.0, 10.0)
}

opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=10, scoring='neg_mean_squared_error', random_state=8)
opt.fit(X_train, y_train)

y_pred = opt.predict(X_test)

# best (minimum) negative mean squared error (MSE) observed during cross-validation
print("Best estimator: ", opt.best_estimator_)
print("Best score: ", opt.best_score_)
print("Test RMSE: ", mean_squared_error(y_test, y_pred, squared=False))

# Feature importance
xgboost_model = opt.best_estimator_.steps[1][1]
plot_importance(xgboost_model)
#plt.show()

def predict_from_input(user_input):
    input_df = pd.DataFrame([user_input])
    
    input_df = input_df[categorical_cols + numeric_cols]
    
    prediction = opt.predict(input_df)
    
    return prediction[0]

def get_user_input():
    user_input = {}
    user_input['storage_type'] = input("Enter storage type (e.g., 'SSD', 'EBS only'): ")
    user_input['instance_name'] = input("Enter instance name (e.g., 't3.xlarge'): ")
    user_input['num_instances'] = int(input("Enter number of instances (e.g., 8): "))
    user_input['vcpu'] = int(input("Enter number of vCPUs (e.g., 4): "))
    user_input['cpu_usage'] = float(input("Enter CPU usage percentage (e.g., 56.431): "))
    user_input['memory'] = float(input("Enter memory in GB (e.g., 16.0): "))
    user_input['memory_usage'] = float(input("Enter memory usage percentage (e.g., 82.9646): "))
    user_input['storage_size'] = float(input("Enter storage size in GB (e.g., 118.7): "))
    return user_input

user_input = get_user_input()

predicted_cost = predict_from_input(user_input)
print("Predicted Cost per Hour:", predicted_cost)