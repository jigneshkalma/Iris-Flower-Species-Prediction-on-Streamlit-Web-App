import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

# For Page Logo
logo_directory = "Images//logo.jpeg"
logo = Image.open(logo_directory)

# For Flower SpeciesImages
Setosa_directory = "SpeciesImages//Setosa.jpeg"
Versicolor_directory = "SpeciesImages//Versicolor.jpeg"
Virginica_directory = "SpeciesImages//Virginica.jpg"
Setosa_image = Image.open(Setosa_directory)
Versicolor_image = Image.open(Versicolor_directory)
Virginica_image = Image.open(Virginica_directory)

# Page Configuration
PAGE_CONFIG = {"page_title": "Iris Flower Species Prediction",
               "page_icon": logo,
               "layout": "centered",
               "initial_sidebar_state": "auto"}

st.set_page_config(**PAGE_CONFIG)


# For Loading Dataset and Storing it into Streamlit Cache Memory
@st.cache_data
def load_data():
    iris = load_iris()
    df1 = pd.DataFrame(iris.data, columns=iris.feature_names)
    df1['species'] = iris.target
    return df1, iris.target_names

df, target_names = load_data()

# Model Training
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

# Slider Creating for User Input
st.sidebar.title("Input Features")
sepal_length = st.sidebar.slider("Sepal length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Prediction
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]
predicted_species = predicted_species.title()

# Prediction Result Displaying to User With Iris Flower Species Image according to Prediction
st.header("Prediction")
st.subheader(f"The predicted species is: {predicted_species}")

if predicted_species == "Setosa":
    st.image(Setosa_image)
elif predicted_species == "Versicolor":
    st.image(Versicolor_image)
elif predicted_species == "Virginica":
    st.image(Virginica_image)
else:
    pass
