#!/usr/bin/env python
# coding: utf-8

#import statements
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer


# Function to preprocess the data
def preprocess_data(df):
    # Preprocess 'Household Income' column
    def preprocess_income(value):
        if 'Above' in value:
            amount = int(value.split('Above US$ ')[1].replace(',', ''))
            return f'{amount + 1000}'
        elif ' to ' in value:
            start, end = value.split(' to ')
            average = (int(start.replace('US$ ', '').replace(',', '')) + int(end.replace('US$ ', '').replace(',', ''))) / 2
            return f'{average}'
        else:
            return value.replace('US$ ', '')
    df['City'] = df['City'].astype('category')
    df['Household Income'] = df['Household Income'].apply(preprocess_income)
    # Now your DataFrame 'df' has label-encoded values for the specified columns
    df['Household Income'] = pd.to_numeric(df['Household Income'], errors='coerce')

    return df

# Function to display intro page
def intro_page():
    st.title("Investment Decision Recommendation System")
    st.write("""
    # Welcome to the Investment Decision Recommendation System . 
    # This system helps in predicting the best investment decision based on various factors. """)


# Function to display about section
def about_section(df):
    st.header("About")
    st.write("Dataset Overview :")
    st.write(df.head())
    st.write("Problem Statement:")
    st.write("You are provided with a dataset containing information about individuals and their investment behavior. Your task is to build a recommendation system that can predict the best investment decision for new data based on various factors available in the dataset.")

#visulaziations
def visualizations(df):
    st.header("Visualizations")

    # Button to display Gender Distribution
    if st.button("Gender Distribution"):
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Gender', data=df, ax=ax)
        plt.title('Distribution of Gender')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        for p in ax.patches:
            ax.annotate(f'\n{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=11, color='black', xytext=(0, 10),
                        textcoords='offset points')
        st.pyplot(fig)

    # Button to display Age Distribution
    if st.button("Age Distribution"):
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.histplot(data=df, x='Age', bins=20, ax=ax)
        plt.title('Distribution of Investors by Age')
        plt.xlabel('Age')
        plt.ylabel('Count')
        st.pyplot(fig)

    # Button to display Investors Based on Gender
    if st.button("Investors Based on Gender"):
        st.subheader("Investors Based on Gender")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='City', hue='Gender', data=df, palette='viridis', ax=ax)
        plt.title('Number of Investors from Different Cities by Gender')
        plt.xlabel('City')
        plt.ylabel('Number of Investors')
        plt.xticks(rotation=45)
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 10),
                        textcoords='offset points',
                        fontsize=10, color='black')
        plt.tight_layout()
        st.pyplot(fig)

    # Button to display Marital Status Distribution
    if st.button("Marital Status Distribution"):
        st.subheader("Marital Status Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Marital Status', hue='Gender', data=df, palette='Set2', ax=ax)
        plt.title('Marital Status Distribution by Gender')
        plt.xlabel('Marital Status')
        plt.ylabel('Count')
        plt.legend(title='Gender')
        st.pyplot(fig)

    # Button to display Relationship between Age and Household Income
    if st.button("Relationship between Age and Household Income"):
        st.subheader("Relationship between Age and Household Income")
        fig, ax = plt.subplots()
        sns.boxplot(x='Age', y='Household Income', data=df, ax=ax)
        plt.title('Relationship between Age and Household Income')
        plt.xlabel('Age')
        plt.ylabel('Household Income')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Button to display Distribution of Return Earned
    if st.button("Distribution of Return Earned"):
        st.subheader("Distribution of Return Earned")
        fig, ax = plt.subplots()
        sns.countplot(x='Return Earned', data=df, palette='viridis', ax=ax)
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
        plt.title('Distribution of Return Earned')
        plt.xlabel('Return Earned')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Button to display Employment Role Distribution by Gender
    if st.button("Employment Role Distribution by Gender"):
        st.subheader("Employment Role Distribution by Gender")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(x='Role', hue='Gender', data=df, palette='viridis', ax=ax)
        plt.title('Employment Role Distribution by Gender')
        plt.xlabel('Role')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Gender')
        st.pyplot(fig)

    # Button to display Risk Level Distribution
 # Button to display Risk Level Distribution
    if st.button("Risk Level Distribution"):
        st.subheader("Risk Level Distribution")
        risk_distribution = df['Risk Level'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(risk_distribution, labels=risk_distribution.index, autopct='%1.1f%%', colors=sns.color_palette('Paired'))
        ax.set_title('Distribution of Risk Levels')
        st.pyplot(fig)


    # Button to display Distribution of Return Earned by Risk Level
    if st.button("Distribution of Return Earned by Risk Level"):
        st.subheader("Distribution of Return Earned by Risk Level")
        fig, ax = plt.subplots()
        sns.countplot(x='Risk Level', hue='Return Earned', data=df, palette='Set2', ax=ax)
        plt.title('Distribution of Return Earned by Risk Level')
        plt.xlabel('Risk Level')
        plt.ylabel('Count')
        plt.legend(title='Return Earned', loc='upper right')
        st.pyplot(fig)

    # Button to display Word Cloud of Investment Reasons
    if st.button("Word Cloud of Investment Reasons"):
        st.subheader("Word Cloud of Investment Reasons")
        investment_reasons = df['Reason for Investment'].dropna().str.cat(sep=', ')
        wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='viridis',
                            max_words=200, contour_color='white', contour_width=1,
                            stopwords=None, prefer_horizontal=0.7, min_font_size=10).generate(investment_reasons)
        fig, ax = plt.subplots(figsize=(10, 6))  
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title('Word Cloud of Investment Reasons', fontsize=20, color='white') 
        ax.axis('off')  
        st.pyplot(fig)  

    # Button to display Percentage of Household Income Invested by Knowledge Level about Investment Products
    if st.button("Percentage of Household Income Invested by Knowledge Level about Investment Products"):
        st.subheader("Percentage of Household Income Invested by Knowledge Level about Investment Products")
        fig, ax = plt.subplots()
        sns.boxplot(x='Knowledge level about different investment product', y='Percentage of Investment', data=df, palette='viridis', ax=ax)
        plt.title('Percentage of Household Income Invested by Knowledge Level about Investment Products')
        plt.xlabel('Knowledge Level about Investment Products')
        plt.ylabel('Percentage of Household Income Invested')
        plt.xticks(rotation=45)
        st.pyplot(fig)

def analysis():
    st.header("Analysis with Power BI Dashboards")
    st.markdown("Gender Distribution")
    st.image("C:/Users/91948/Desktop/Buckman/dashboard1.jpg", use_column_width=True)

    st.markdown("City Distribution")
    st.image("C:/Users/91948/Desktop/Buckman/dashboard2.jpg", use_column_width=True)
    
    st.markdown("Return Earned Distribution")
    st.image("C:/Users/91948/Desktop/Buckman/dashboard3.jpg", use_column_width=True)

    st.markdown("Knowledge about Investment ")
    st.image("C:/Users/91948/Desktop/Buckman/dashboard4.jpg", use_column_width=True)
    
    st.markdown("HouseHold Income")
    st.image("C:/Users/91948/Desktop/Buckman/dashboard5.jpg", use_column_width=True)
    
    
# Function to build and evaluate models
def inference(df):
    label_encoder = LabelEncoder()
    columns_to_encode = ['City', 'Gender', 'Marital Status', 'Education', 'Age', 'Role', 'Percentage of Investment', 'Risk Level', 'Source of Awareness about Investment', 'Investment Experience', 'Investment Influencer', 'Risk Level', 'Return Earned', 'Reason for Investment']
    for col in columns_to_encode:
        df[col] = label_encoder.fit_transform(df[col])

    # Correlation Matrix
# Correlation Matrix
    if st.button("CORRELATION MATRIX"):
        st.subheader("CORRELATION MATRIX")
        correlation_matrix = df.corr()
        fig, ax = plt.subplots(figsize=(12, 8)) 
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Matrix') 
        st.pyplot(fig) 


    # Model building
    X = df.drop(columns=['Return Earned'])
    y = df['Return Earned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    y_pred_dt = dt_classifier.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    report_dt = classification_report(y_test, y_pred_dt)

    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf)

    '''# Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)
    y_pred_rfs = rf_classifier.predict(X_test_scaled)
    accuracy_rfs = accuracy_score(y_test, y_pred_rfs)
    report_rfs = classification_report(y_test, y_pred_rfs)
    
    rf_classifier = RandomForestClassifier(random_state=42)
   
    #Hyperparameters to tune
    param_grid = {
        'n_estimators': [100, 200, 300],  
        'max_depth': [None, 10, 20],  
        'min_samples_split': [2, 5, 10], 
        'min_samples_leaf': [1, 2, 4],  
        'max_features': ['auto', 'sqrt'],  
        'bootstrap': [True, False]  
    }
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_grid,n_iter=100, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    y_pred_tune = best_model.predict(X_test)
    test_accuracy_tune = accuracy_score(y_test, y_pred_tune)

    #K-fold Cross Validition
    from sklearn.ensemble import RandomForestClassifier 
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    #k=7
    cv_scores = cross_val_score(model, X, y, cv=7)
    mean_cv_score = cv_scores.mean()
    std_cv_score = cv_scores.std()'''

    # MLP Classifier - Neural Network
    mlp = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='adam', max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_nn = mlp.predict(X_test)
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    report_nn = classification_report(y_test, y_pred_nn)

    # Gradient Boosting Classifier
    gb_classifier = GradientBoostingClassifier()
    gb_classifier.fit(X_train, y_train)
    y_pred_gb = gb_classifier.predict(X_test)
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    report_gb = classification_report(y_test, y_pred_gb)


    st.header("Model Evaluation")
        
    st.write(f"Decision Tree Classifier Accuracy: {accuracy_dt}")
    st.write(f"Random Forest Classifier Accuracy: {accuracy_rf}")
    st.write(f"MLP Classifier Accuracy: {accuracy_nn}")
    #st.write(f"Accuracy using Standard scaler on Random Forest: {accuracy_rfs:.2f}")
    #st.write("Test Accuracy (Best Model) after Parameter tuning on RF:", test_accuracy_tune)
    #st.write("Mean Cross-Validation Score on RF:", mean_cv_score)
    st.write("Best Accuracy (Gradient Boosting):", accuracy_gb)


    st.write("Decision Tree Classifier Classification Report:\n", report_dt)
    st.write("Random Forest Classifier Classification Report:\n", report_rf)
    st.write("MLP Classifier Classification Report:\n", report_nn)
    st.write("Gradient Boosting Classifier Classification Report:\n", report_gb)

def predict(df):
    # Preprocess the input data
    st.subheader("Enter Data for Prediction")

    # Collect input data
    city = st.text_input("City")
    gender = st.selectbox("Gender", ["Men", "Women"])
    marital_status = st.selectbox("Marital Status", ["Never Married", "Married", "Divorced", "Widowed"])
    age = st.selectbox("Age", ["Early Working", "Young Adult", "Middle-Aged", "Senior"])
    education = st.selectbox("Education", ["Secondary", "Higher Secondary", "Graduate", "Postgraduate"])
    role = st.text_input("Role")
    investors_in_family = st.number_input("Number of investors in family")
    household_income = st.text_input("Household Income")
    percentage_investment = st.text_input("Percentage of Investment")
    awareness_source = st.selectbox("Source of Awareness about Investment", ["Television", "Newspaper", "Internet", "Friends/Family", "Other"])
    knowledge_investment = st.slider("Knowledge level about different investment product", min_value=0, max_value=10, value=5)
    knowledge_sharemarket = st.slider("Knowledge level about sharemarket", min_value=0, max_value=10, value=5)
    knowledge_schemes = st.slider("Knowledge about Govt. Schemes", min_value=0, max_value=10, value=5)
    investment_influencer = st.selectbox("Investment Influencer", ["Family Reference", "Financial Advisor", "Online Research", "Other"])
    investment_experience = st.selectbox("Investment Experience", ["Less Than 1 Year", "1-3 Years", "3-5 Years", "More Than 5 Years"])
    risk_level = st.selectbox("Risk Level", ["Low", "Medium", "High"])
    reason_investment = st.text_input("Reason for Investment")
    
    if st.button("Predict"):
        input_data = pd.DataFrame({
            'City': [city],
            'Gender': [gender],
            'Marital Status': [marital_status],
            'Age': [age],
            'Education': [education],
            'Role': [role],
            'Number of investors in family': [investors_in_family],
            'Household Income': [household_income],
            'Percentage of Investment': [percentage_investment],
            'Source of Awareness about Investment': [awareness_source],
            'Knowledge level about different investment product': [knowledge_investment],
            'Knowledge level about sharemarket': [knowledge_sharemarket],
            'Knowledge about Govt. Schemes': [knowledge_schemes],
            'Investment Influencer': [investment_influencer],
            'Investment Experience': [investment_experience],
            'Risk Level': [risk_level],
            'Reason for Investment': [reason_investment]
        })

        input_data = preprocess_data(input_data)
        input_data = input_data.astype(str)
        # Label encode the categorical columns
        
        label_encoder = LabelEncoder()
        columns_to_encode = ['City', 'Gender', 'Marital Status', 'Education', 'Age', 'Role', 'Percentage of Investment', 'Risk Level', 'Source of Awareness about Investment', 'Investment Experience', 'Investment Influencer', 'Risk Level', 'Reason for Investment']
        for col in columns_to_encode:
            input_data[col] = label_encoder.fit_transform(input_data[col])

        print(input_data)
        # Load the trained RandomForestClassifier model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_train = df.drop(columns=['Return Earned'])
        y_train = df['Return Earned']
        rf_model.fit(X_train, y_train)

        # Use the model for prediction
        prediction = rf_model.predict(input_data)[0]

        st.success(f"The predicted return earned is: {prediction}")

#main function
def main():
    df = pd.read_excel("C:/Users/91948/Desktop/Buckman/Sample Data for shortlisting.xlsx")
    df = preprocess_data(df)
    # Preprocess the dataset
    # Now your DataFrame 'df' has label-encoded values for the specified columns
    intro_page()
    option = st.sidebar.selectbox('Select Option', ('About', 'Visualizations', 'Inference', 'Predict','Analysis'))

    # Based on the selected option, display the corresponding section
    if option == 'About':
        about_section(df)
    elif option == 'Visualizations':
        visualizations(df)  # Pass the preprocessed DataFrame for visualizations
    elif option == 'Inference':
        inference(df)  # Pass the preprocessed DataFrame for inference
    elif option == 'Predict':
        predict(df)
    elif option =='Analysis':
        analysis()

# Run the app
if __name__ == '__main__':
    main()
