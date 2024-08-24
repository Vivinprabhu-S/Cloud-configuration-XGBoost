import pandas as pd

#dataset
cloud_data = pd.read_csv('./cloud.csv')


# number of rows and columns
print("Number of rows and columns")
print(cloud_data.shape)

# dataset first 5 data
print("First 5 data")
print(cloud_data.head())

# dataset last 5 data
print("Last 5 data")
print(cloud_data.tail())


# selecting the relevant features for recommendation

selected_features = ['numberOfInstances','instanceName','storageSize','costPerHour']
print(selected_features)


#fill the null values with ''
for feature in selected_features:
  cloud_data[feature] = cloud_data[feature].fillna('')

# combining all the 5 selected features
combined_features = (
    cloud_data['numberOfInstances'].astype(str) + ' ' +
    cloud_data['instanceName'] + ' ' +
    cloud_data['storageSize'].astype(str) + ' ' +
    cloud_data['costPerHour'].astype(str) + ' ' 
    )

print(combined_features)


# converting the text data to feature vectors

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
print(feature_vectors)



# getting the similarity scores using cosine similarity

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(feature_vectors)

print(similarity)
print("Number of rows and columns after cosine_similarity:")
print(similarity.shape)


cur_num_instances = input('Enter the number of instances: ')
cur_instance_name = input('Enter the instance name: ')
cur_storage_size = input('Enter the storage size: ')
cur_cost_per_hour = input('Enter the cost per hour: ')

# Combine user inputs into a single string
user_input = (
    cur_num_instances + ' ' +
    cur_instance_name + ' ' +
    cur_storage_size + ' ' +
    cur_cost_per_hour
)

# Vectorize the user input
user_input_vector = vectorizer.transform([user_input])

# Compute similarity between the user input and the dataset
user_similarity = cosine_similarity(user_input_vector, feature_vectors)

# Get a list of similarity scores
similarity_score = list(enumerate(user_similarity[0]))
#print(similarity_score)

# Sort the recommendations based on similarity scores
sorted_similar = sorted(similarity_score, key=lambda x: x[1], reverse=True)
#print(sorted_similar)

print('Recommendations for you with lower cost per hour:\n')

i = 1
cur_cost_per_hour = float(cur_cost_per_hour)  

for instance in sorted_similar:
    index = instance[0]
    num_instances = cloud_data.loc[index, 'numberOfInstances']
    instance_name = cloud_data.loc[index, 'instanceName']
    storage_size = cloud_data.loc[index, 'storageSize']
    cost_per_hour = cloud_data.loc[index, 'costPerHour']
    
    if float(cost_per_hour) < cur_cost_per_hour:
        if i <= 10:
            print(f"{i}. Number of Instances: {num_instances}, Instance Name: {instance_name}, Storage Size: {storage_size}, Cost Per Hour: {cost_per_hour}")
            i += 1

# If no recommendations are found
if i == 1:
    print("Your current plan is already economically good.")