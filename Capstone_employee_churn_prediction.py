import streamlit as st
import pickle
import pandas as pd


st.sidebar.title('Employee Churn Analysis')


html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Employee Churn Analysis Prediction with Random Forest Model</h2>
</div><br><br>"""


st.markdown(html_temp,unsafe_allow_html=True)

selection=st.selectbox("Select Your Model", ["KNN", "Random Forest"])

if selection =="KNN":
	st.write("You selected", selection, "model")
	model= pickle.load(open('model_proje_2KNN', 'rb'))
else:
	st.write("You selected", selection, "model")
	model= pickle.load(open('model_proje_2', 'rb'))
    

satisfaction_level=st.sidebar.selectbox("employee satisfaction point",(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
last_evaluation=st.sidebar.selectbox("evaluated performance by the employer", (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
number_project=st.sidebar.selectbox("how many of projects assigned to an employee", (1,2,3,4,5,6,7))
average_montly_hours=st.slider("how many hours in averega an employee worked in a month", 0, 310, 96, step=1)
time_spend_company=st.sidebar.selectbox("the number of years spent by an employee in the company", (1,2,3,4,5,6,7,8,9,10))
Work_accident=st.sidebar.selectbox("work_accident",(0,1))
promotion_last_5years=st.sidebar.selectbox("promotion_last_5years",(0,1))
salary_new=st.sidebar.selectbox("salary level",(0,1,2))


my_dict={'satisfaction_level':satisfaction_level, 
        'last_evaluation':last_evaluation, 
        'number_project':number_project,
        'average_montly_hours':average_montly_hours,
        'time_spend_company':time_spend_company,
        'Work_accident':Work_accident, 
        'promotion_last_5years':promotion_last_5years,
        'salary_new':salary_new}


columns=['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'salary_new']

df = pd.DataFrame.from_dict([my_dict])

X = pd.get_dummies(df).reindex(columns=columns, fill_value=0)

prediction = model.predict(X)

st.header("Your informations are below")
st.table(df)
st.subheader("Press predict if your informations are okay")

if st.button('Predict'):
    st.success("The  employee loss analysis estimated {}. ".format(int(prediction[0])))



     
   
    





    


