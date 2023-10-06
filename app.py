import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset (you can adjust the path accordingly)
df = pd.read_csv('dataset.csv')

# Separate features (salary and family member) and target (zone)
X = df[['Salary', 'Member']]  # Use 'Member' here
y = df['Zone']

# Standardize features (scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the K-NN classifier with a chosen value of k (e.g., k=3)
k = 9
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the model on the entire dataset
knn_classifier.fit(X_scaled, y)

# Streamlit UI
st.title('K-NN Classifier')

# Input form
st.write("Enter Salary:")
salary = st.number_input("Salary", min_value=0.0)
st.write("Enter Family Members:")
family_member = st.number_input("Family Members", min_value=0)

if st.button('Predict'):
    # Create a DataFrame with the user input
    user_data = pd.DataFrame({'Salary': [salary], 'Member': [family_member]})  # Use 'Member' here

    # Scale the user input data using the same scaler
    user_data_scaled = scaler.transform(user_data)

    # Find the k-nearest neighbors for the user input
    distances, indices = knn_classifier.kneighbors(user_data_scaled, n_neighbors=k)

    # Get the predicted zone for the user input
    predicted_zone = knn_classifier.predict(user_data_scaled)[0]

    # Get the zones of the k-nearest neighbors
    neighbor_zones = [y.iloc[i] for i in indices[0]]

    # Display the prediction result and neighbors
    st.write(f"The predicted zone for the given input is: {predicted_zone}")
    st.subheader(f'{k} Nearest Neighbors:')
    for i, (zone, distance) in enumerate(zip(neighbor_zones, distances[0]), start=1):
        st.write(f'Neighbor {i}: Zone {zone}, Distance: {round(distance, 2)}')

    st.button('Back to Input')
