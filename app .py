
"""
Construction Material Cost Prediction - Streamlit App
Requirements:
streamlit
pandas
numpy
scikit-learn
matplotlib
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Construction Material Cost Prediction", layout="wide")

@st.cache_data
def load_data():
    return pd.DataFrame({
        "Building_Area":[120,150,180,200,220,250,280,300,330,350,380,420],
        "Cement_Bags":[320,410,520,610,670,760,840,920,990,1080,1160,1280],
        "Steel_Tons":[8,10,12,15,17,20,22,25,27,30,33,36],
        "Sand_m3":[45,60,75,82,90,110,120,135,145,160,175,190],
        "Labour":[20,24,30,34,38,42,45,48,52,55,60,65],
        "Material_Cost":[135000,175000,225000,265000,295000,345000,385000,435000,475000,520000,570000,630000]
    })

df=load_data()
st.title("Construction Material Cost Prediction")
tabs=st.tabs(["Overview","EDA","Prediction","Model Results"])

X=df[["Building_Area","Cement_Bags","Steel_Tons","Sand_m3","Labour"]]
y=df["Material_Cost"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression().fit(X_train,y_train)
pred=model.predict(X_test)
mae=mean_absolute_error(y_test,pred)
rmse=mean_squared_error(y_test,pred)**0.5
r2=r2_score(y_test,pred)

with tabs[0]:
    st.write("Predict construction material cost using Linear Regression.")
    st.dataframe(df)

with tabs[1]:
    fig,ax=plt.subplots()
    ax.scatter(df["Building_Area"],df["Material_Cost"])
    ax.set_xlabel("Building Area")
    ax.set_ylabel("Material Cost")
    st.pyplot(fig)

with tabs[2]:
    area=st.number_input("Building Area",100,1000,240)
    cement=st.number_input("Cement Bags",100,5000,700)
    steel=st.number_input("Steel (tons)",1,100,18)
    sand=st.number_input("Sand (m3)",1,500,100)
    labour=st.number_input("Labour",1,500,40)
    if st.button("Predict"):
        new=pd.DataFrame({"Building_Area":[area],"Cement_Bags":[cement],"Steel_Tons":[steel],"Sand_m3":[sand],"Labour":[labour]})
        cost=model.predict(new)[0]
        st.success(f"Estimated Material Cost: Nu. {cost:,.2f}")

with tabs[3]:
    c1,c2,c3=st.columns(3)
    c1.metric("MAE",f"{mae:,.2f}")
    c2.metric("RMSE",f"{rmse:,.2f}")
    c3.metric("R²",f"{r2:.3f}")
    st.write(pd.DataFrame({"Actual":y_test.values,"Predicted":pred}))
    import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def load_data():
    # Example dataset (replace with your actual data)
    data = pd.DataFrame({
        "Material": ["Cement", "Steel", "Bricks", "Sand"],
        "Cost": [5000, 12000, 3000, 2000]
    })
    return data

# Load data
df = load_data()

st.title("Construction Material Cost Prediction")

# Show data table
st.write("### Dataset Preview")
st.dataframe(df)

# Create a bar chart
fig, ax = plt.subplots()
ax.bar(df["Material"], df["Cost"], color="skyblue")
ax.set_xlabel("Material")
ax.set_ylabel("Cost")
ax.set_title("Material Cost Comparison")

# Display chart in Streamlit
st.pyplot(fig)

