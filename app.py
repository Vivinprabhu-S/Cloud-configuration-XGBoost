from flask import Flask, render_template, request
import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
cloud_data = pd.read_csv('./cloud.csv')

# Recommendation: Preprocess and train models for recommendation
selected_features = ['numberOfInstances', 'instanceName', 'storageSize', 'costPerHour']
cloud_data[selected_features] = cloud_data[selected_features].fillna('')
combined_features = (
    cloud_data['numberOfInstances'].astype(str) + ' ' +
    cloud_data['instanceName'] + ' ' +
    cloud_data['storageSize'].astype(str) + ' ' +
    cloud_data['costPerHour'].astype(str)
)
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Forecasting: Preprocess and train models for cost forecasting
df = cloud_data.rename(columns={
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

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', TargetEncoder(), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', XGBRegressor(random_state=8)) 
])
pipe.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    recommendations = []
    if request.method == 'POST':
        cur_num_instances = request.form['numberOfInstances']
        cur_instance_name = request.form['instanceName']
        cur_storage_size = request.form['storageSize']
        try:
            cur_cost_per_hour = float(request.form['costPerHour'])
        except ValueError:
            return render_template('index.html', recommendations=[{'message': 'Invalid cost per hour. Please enter a valid number.'}])

        user_input = (
            cur_num_instances + ' ' +
            cur_instance_name + ' ' +
            cur_storage_size + ' ' +
            str(cur_cost_per_hour)
        )
        user_input_vector = vectorizer.transform([user_input])
        user_similarity = cosine_similarity(user_input_vector, feature_vectors)
        similarity_score = list(enumerate(user_similarity[0]))
        sorted_similar = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        i = 1
        for instance in sorted_similar:
            index = instance[0]
            num_instances = cloud_data.loc[index, 'numberOfInstances']
            instance_name = cloud_data.loc[index, 'instanceName']
            storage_size = cloud_data.loc[index, 'storageSize']
            cost_per_hour = float(cloud_data.loc[index, 'costPerHour'])

            if cost_per_hour < cur_cost_per_hour:
                if i <= 10:
                    recommendations.append({
                        'Number of Instances': num_instances,
                        'Instance Name': instance_name,
                        'Storage Size': storage_size,
                        'Cost Per Hour': cost_per_hour
                    })
                    i += 1

        if not recommendations:
            recommendations.append({'message': 'Your current plan is already economically good.'})

    return render_template('index.html', recommendations=recommendations)

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    predicted_cost = None
    if request.method == 'POST':
        user_input = {
            'storage_type': request.form['storage_type'],
            'instance_name': request.form['instance_name'],
            'num_instances': int(request.form['num_instances']),
            'vcpu': int(request.form['vcpu']),
            'cpu_usage': float(request.form['cpu_usage']),
            'memory': float(request.form['memory']),
            'memory_usage': float(request.form['memory_usage']),
            'storage_size': float(request.form['storage_size'])
        }

        input_df = pd.DataFrame([user_input])[categorical_cols + numeric_cols]
        predicted_cost = pipe.predict(input_df)[0]

    return render_template('forecast.html', predicted_cost=predicted_cost)

if __name__ == '__main__':
    app.run(debug=True)
