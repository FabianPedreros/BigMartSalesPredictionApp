# Big Mart Sales prediction

# Import all libreries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Streamlit

st.set_page_config(layout='wide', initial_sidebar_state='collapsed')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Show the title in the app
st.markdown('# *:red[Big Mart Sales Prediction]*')

st.markdown('## :blue[1. Introduction]')

st.markdown("""In this project we are going to apply the IBM - Jonh Rollings methodology for data science, 
with the goal of generate a model that can be used to predict the sales for the next period, as accurately as possible.

Therefore, in this application we are going to see all the steps that are needed to solve a data science problem and finally create a dashboard where a user can 
interact with the data and the models. We will then move step by step through each component of the methodology, from 'Business understanding' to 'Deployment'

This problem has been taken from the practice problem formulated by [Analytics Vidhya](https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/)
and the data has been downloaded from the Kaggle data set [Big Mart Sales Prediction Datasets
](https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-datasets)""")


st.markdown('## :blue[2. Methodology]')

st.markdown(""" We are going to use a methodology created by John Rollings and actively used by IBM in the solution and implementation of data science projects. 
This methodology aims to establish a series of activities to solve an analytical problem, proposing ten possible stages that must be solved in any data science project.
""")


st.markdown('This metodology aims to answer ten different questions.')

st.image('Methodology.PNG', width = 500, caption = 'Data Science Methodology - IBM')
st.image('IBMSteps.png', width = 500, caption = 'Steps - IBM')

st.markdown('## :blue[3. Theoretical framework]')

st.markdown("""**Documents and projects consulted**
- [Your Guide to Linear Regression Models](https://www.kdnuggets.com/2020/10/guide-linear-regression-models.html) - Diego Lopez Yse - KDnuggets.
- [14 Different Types of Learning in Machine Learning](https://machinelearningmastery.com/types-of-learning-in-machine-learning/) - Jason Brownlee - machinelearningmastery.com
- [6 Types of Regression Models in Machine Learning You Should Know About](https://www.upgrad.com/blog/types-of-regression-models-in-machine-learning/) - Pavan Vadapalli - upgrad.com
- [Approach and Solution to break in Top 20 of Big Mart Sales prediction](https://www.analyticsvidhya.com/blog/2016/02/bigmart-sales-solution-top-20/) - Aarshay Jain - Analytics Vidhya
""")

st.markdown('### :blue[Machine learning]')
st.markdown("""Machine learning is a large field of study that overlaps with and inherits ideas from many related fields such as artificial intelligence. 
The focus of the field is learning, that is, acquiring skills or knowledge from experience. Most commonly, this means synthesizing useful concepts from historical data. In this case scenario we are going to 
learn from sales historical data in order to predict the future sales of every single product and store.
""")

st.markdown('### :blue[Supervised learning]')
st.markdown("""Describes a class of problem that involves using a model to learn a mapping between input examples and the target variable.
Models are fit on training data comprised of inputs and outputs and used to make predictions on test sets where only the inputs are provided and the outputs from the model are compared to the withheld target variables and used to estimate the skill of the model.


There are two main types of supervised learning problems: they are classification that involves predicting a class label and regression that involves predicting a numerical value.

- **Classification:** Supervised learning problem that involves predicting a class label.
- **Regression:** Supervised learning problem that involves predicting a numerical label.
Both classification and regression problems may have one or more input variables and input variables may be any data type, such as numerical or categorical.

An example of a regression problem would be the Boston house prices dataset where the inputs are variables that describe a neighborhood and the output is a house price in dollars.
""")

st.markdown('### :blue[Regression]')
st.markdown("""Regression analysis is a predictive modelling technique that analyzes the relation between the target or dependent variable and independent variable in a dataset. 
The different types of regression analysis techniques get used when the target and independent variables show a linear or non-linear relationship between each other, and the target variable contains continuous values. 
The regression technique gets used mainly to determine the predictor strength, forecast trend, time series, and in case of cause & effect relation. 

Regression analysis is the primary technique to solve the regression problems in machine learning using data modelling. 
It involves determining the best fit line, which is a line that passes through all the data points in such a way that distance of the line from each data point is minimized.

The different types of regression in machine learning techniques are explained below in detail:

#### 1. Linear Regression
Linear regression is one of the most basic types of regression in machine learning. The linear regression model consists of a predictor variable and a dependent variable related linearly to each other. In case the data involves more than one independent variable, then linear regression is called multiple linear regression models. 

The below-given equation is used to denote the linear regression model:
""")

st.latex('y=mx+c+e')
st.markdown('where m is the slope of the line, c is an intercept, and e represents the error in the model.')
st.image('https://www.upgrad.com/blog/wp-content/uploads/2020/07/438px-Linear_regression.svg-1.png', width = 300, caption='Linear regression')
st.markdown("""The best fit line is determined by varying the values of m and c. The predictor error is the difference between the observed values and the predicted value. The values of m and c get selected in such a way that it gives the minimum predictor error. It is important to note that a simple linear regression model is susceptible to outliers. 
Therefore, it should not be used in case of big size data.
There are different types of linear regression. The two major types of linear regression are simple linear regression and multiple linear regression. Below is the formula for simple linear regression.
""")

st.markdown("""#### 2. Ridge regression
This is another one of the types of regression in machine learning which is usually used when there is a high correlation between the independent variables. This is because, in the case of multi collinear data, the least square estimates give unbiased values. But, in case the collinearity is very high, there can be some bias value.
Therefore, a bias matrix is introduced in the equation of Ridge Regression. This is a powerful regression method where the model is less susceptible to overfitting. 
""")
st.image('https://www.upgrad.com/blog/wp-content/uploads/2020/07/ridge_regression_geomteric-1.png', width = 300, caption= 'Ridge regression')
st.markdown('Below is the equation used to denote the Ridge Regression, where the introduction of λ (lambda) solves the problem of multicollinearity:')
st.latex('β = (X^{T}X + λ*I)^{-1}X^{T}y')

st.markdown(""" #### 3. Lasso regression
Lasso Regression is one of the types of regression in machine learning that performs regularization along with feature selection. It prohibits the absolute size of the regression coefficient. As a result, the coefficient value gets nearer to zero, which does not happen in the case of Ridge Regression.

Due to this, feature selection gets used in Lasso Regression, which allows selecting a set of features from the dataset to build the model. In the case of Lasso Regression, only the required features are used, and the other ones are made zero. This helps in avoiding the overfitting in the model.
In case the independent variables are highly collinear, then Lasso regression picks only one variable and makes other variables to shrink to zero.
""")
st.image('https://www.upgrad.com/blog/wp-content/uploads/2020/07/lasso-reg-coef-1.png', width = 300, caption= 'Regression coefficients Lasso regression')
st.markdown('Below is the equation that represents the Lasso Regression method:')
st.latex('N^{-1}Σ^{N}_{i=1}f(x_{i}, y_{I}, α, β)')

st.markdown("""#### 4. Polynomial regression
Polynomial Regression is another one of the types of regression analysis techniques in machine learning, which is the same as Multiple Linear Regression with a little modification. In Polynomial Regression, the relationship between independent and dependent variables, that is X and Y, is denoted by the n-th degree.

It is a linear model as an estimator. Least Mean Squared Method is used in Polynomial Regression also. The best fit line in Polynomial Regression that passes through all the data points is not a straight line, but a curved line, which depends upon the power of X or value of n.
""")
st.image('https://www.upgrad.com/blog/wp-content/uploads/2020/07/341px-Polyreg_scheffe.svg-1.png', width = 300, caption = 'Polynomial regression')

st.markdown("""While trying to reduce the Mean Squared Error to a minimum and to get the best fit line, the model can be prone to overfitting. It is recommended to analyze the curve towards the end as the higher Polynomials can give strange results on extrapolation. 

Below equation represents the Polynomial Regression:""")

st.latex('l = β0+ β0x1+ε')

st.markdown('***Taken from:*** [6 Types of Regression Models in Machine Learning You Should Know About](https://www.upgrad.com/blog/types-of-regression-models-in-machine-learning/) - Pavan Vadapalli - upgrad.com')

st.markdown('With all these concepts in mind, we now begin with the use of the methodology applied to the scenario, from here we will develop each stage of the methodology to solve the questions posed.')

st.markdown('## :blue[4. Business understanding]')
st.markdown("""Establish the problem to be solved:

***- What is the problem you are trying to solve?***

We are goiong to work with the same problem statement formulated for Analytics Vidhya:

The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. ***The aim is to build a predictive model and find out the sales of each product at a particular store.***

Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.
""")

st.markdown('## :blue[5. Analytic approach]')

st.markdown("""Identify types of patterns needed to solve the question effectively:

***- How can you use data to answer the question?***

Since we have to give a continues number that is the sales prediction for every product and store, and we have historical data with attributes of each product and store, we are going to train different unsupervised regression algorithms, and then perform a model evaluation to select the one with the highest accuracy.

The algorithms we will train and evaluate are:

- Linear regression
- Ridge regression
- Lasso regression
- Polynomial regression
""")


st.markdown('## :blue[6. Data requirements]')
st.markdown("""Identify required data, formats and sources

***- What data do you need to answer the question?***

The data we are going to use is the provided in Kaggle for Shivan Kumar used in the Analytics Vidhya hackaton.
These data set has three csv files, one with the training data, another with the test data and the third contains the submission file (dependent variable and data items for the test data)

Some meta data that describes the independent and dependent varaibles:

- **ItemIdentifier:** Unique product ID
- **ItemWeight:** Weight of product
- **ItemFatContent:** Whether the product is low fat or not
- **ItemVisibility:** The % of the total display area of all products in a store allocated to the particular product
- **ItemType:** The category to which the product belongs
- **ItemMRP:** Maximum Retail Price (list price) of the product
- **OutletIdentifier:** Unique store ID
- **OutletEstablishmentYear:** The year in which the store was established
- **OutletSize:** The size of the store in terms of ground area covered
- **OutletLocationType:** The type of city in which the store is located
- **OutletType:** Whether the outlet is just a grocery store or some sort of supermarket
- **ItemOutletSales:** sales of the product in t particular store. This is the outcome variable to be predicted.
""")

st.markdown('## :blue[7. Data collection]')
st.markdown("""Data collection and validation

***- Where is the data coming from and how will you get it?***

The data used are taken from the data set [Big Mart Sales Prediction Datasets](https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-datasets) in  Kaggle.
This data is downloaded in form of three csv files and savedit in a local repository. 
""")

st.markdown('## :blue[8. Data understanding]')
st.markdown("""Establish whether the data are representative of the problem to be solved. Develop an **Exploratory Data Analysis - EDA**.

- ***Is the data that you collected representative of the problem to be solved?***

We will develop an EDA to understand the data and validate that it is sufficient to model the regression algorithms. Due the data is divide in three files the first step is to join all the data.
""")

st.markdown('## :blue[8.1. Import libraries and data reading]')

# Import the required libreries for the EDA
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings
import io
warnings.filterwarnings('ignore')

#path = r'E:\Estudio\Análisis de datos\Proyectos\Big Mart Sales prediction/'

st.code("""# Import the required libreries for the EDA
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

path = r'E:\Estudio\Análisis de datos\Proyectos\Big Mart Sales prediction/'""")

@st.cache(allow_output_mutation=True)
def read_csv():
    # Reading data from each csv file. 
    train_df = pd.read_csv('train.csv', sep=',')
    test_df = pd.read_csv('test.csv', sep=',')
    submission_df = pd.read_csv('sample_submission.csv', sep=',')
    return train_df, test_df, submission_df

train_df, test_df, submission_df = read_csv()

st.code("""# Reading data from each csv file. 
train_df = pd.read_csv(path + 'train.csv', sep=',')
test_df = pd.read_csv(path + 'test.csv', sep=',')
submission_df = pd.read_csv(path + 'sample_submission.csv', sep=',')""")


st.code("""# Visualizing the dataframes
train_df.head()""")

# Visualizing the dataframes
st.dataframe(train_df.head())

st.code('test_df.head()')
st.dataframe(test_df.head())

st.code('submission_df.head()')
st.dataframe(submission_df)


st.markdown('Review the dataframes shapes')
st.code("print(f'The trainig data has: {train_df.shape[0]} rows and {train_df.shape[1]} columns')")
st.write('The trainig data has:', train_df.shape[0], 'rows and ', train_df.shape[1], 'columns')

st.code("print(f'The test data has: {test_df.shape[0]} rows and {test_df.shape[1]} columns')")
st.write('The test data has:', test_df.shape[0], 'rows and ', test_df.shape[1], 'columns')

st.code("print(f'The submission data has: {submission_df.shape[0]} rows and {submission_df.shape[1]} columns')")
st.write('The submission data has:', submission_df.shape[0], 'rows and ', submission_df.shape[1], 'columns')

st.markdown('- Merge the three dataframes in one.')

# Merge the three dataframes in one.
# Join the dependent variable with the test data set.
df_test_sub = pd.merge(test_df, submission_df, how='inner', on= ['Item_Identifier', 'Outlet_Identifier'])
st.code("""# Join the dependent variable with the test data set.
df_test_sub = pd.merge(test_df, submission_df, how='inner', on= ['Item_Identifier', 'Outlet_Identifier'])""")
st.dataframe(df_test_sub)

# Merge the two remaining dataframes
df_all = train_df.append(df_test_sub)
st.code("""# Merge the two remaining dataframes
df_all = train_df.append(df_test_sub)""")
st.dataframe(df_all)

st.markdown('Validate the dataframe shape and the column names')
st.code("print(f'The data has: {df_all.shape[0]} rows and {df_all.shape[1]} columns')")
st.write('The data has:', df_all.shape[0], 'rows and ', df_all.shape[1], 'columns')

st.code("print(f'The column names are: {df_all.columns}')")
st.write('The column names are:', df_all.columns)
st.markdown(""":red[The dataframe is ready to work with it, but if we check the dependent data of the submission dataset (Item_Outlet_Sales) we can see how we have a fixed value of 1000 for all 5681 rows, 
because of this we cannot work with the test and submission data so the following steps we are going to perform with the train dataset.]

:red[To work with the available data and to avoid misunderstandings, we will change the name of train_df to df.]
""")

df=train_df.copy()
st.code('df=train_df.copy()')
st.dataframe(df)


st.markdown('## :blue[8.2. Data Cleansing]')

st.markdown('Obtain initial information from the data set')
st.code('df.info()')
buffer=io.StringIO()
df.info(buf = buffer)
s=buffer.getvalue()
st.text(s)
st.markdown("""From this info we can see how the attributes have the correct initial item type, categorical attributes are objects and the numerical are integer or float

We can also see the existence of null values in some attributes such as: Item_Weight and Outlet_Size, so we are going to validate the null quantities and their percentage, and thus determine a good method to deal with missing values.
""")
st.markdown('- Null values quantities')
st.code('df.isnull().sum()')
st.text(df.isnull().sum())

st.markdown('- Null values percentage')
st.code('df.isnull().sum()*100/df.shape[0]')
st.text(df.isnull().sum()*100/df.shape[0])

st.markdown("""Now that we know the attributes with missing values their quantities and percentages, we need to validate for the categorical attributes what are the distributions in the categories and for the qualitative attributes the central statistics,
in order to establish which imputation method to use.""")


st.markdown("""I find out that many of the rows that dont have item weight are from items that have a weight in another row, and the items that have weight is consisten in all the rows where it is assigned,
so i assigned the item weight to the ones that have it in null.""")


st.dataframe(df['Item_Identifier'][df['Item_Weight'].isnull()].unique())
st.dataframe(df[['Item_Identifier','Item_Weight']][df['Item_Weight'].notnull()].drop_duplicates('Item_Identifier'))

a = df['Item_Identifier'][df['Item_Weight'].isnull()].unique()
#b = df[['Item_Identifier','Item_Weight']][df['Item_Weight'].notnull()].drop_duplicates('Item_Identifier')
b = df['Item_Identifier'][df['Item_Weight'].notnull()].unique()

st.markdown('Items that in some rows have a null weight but it is present in another rows')
st.write([i for i in a if i in b])

c = df[['Item_Identifier','Item_Weight']][df['Item_Weight'].notnull()].drop_duplicates('Item_Identifier')
#st.write(c.shape)

d = df[['Item_Identifier','Item_Weight']][df['Item_Weight'].notnull()].drop_duplicates(['Item_Identifier','Item_Weight'])
#st.write(d.shape)

df = pd.merge(c, df, how='right', on ='Item_Identifier')
df.drop(['Item_Weight_y'], axis = 1, inplace= True)
df.rename(columns={'Item_Weight_x':'Item_Weight'}, inplace=True)
st.markdown('df with the weights association')
st.dataframe(df)

st.markdown('- Now that we have a low null values for weight, we can make a median imputation')
st.code('df.isnull().sum()*100/df.shape[0]')
st.text(df.isnull().sum()*100/df.shape[0])

st.dataframe(df.describe().iloc[:,:1])

df['Item_Weight'].fillna(np.mean(df['Item_Weight']), inplace=True)
st.code("df['Item_Weight'].fillna(np.mean(df['Item_Weight']), inplace=True)")

st.markdown('- Results after mean imputation for Item_Weight')
st.text(df.isnull().sum()*100/df.shape[0])

st.markdown("As we can see, the Outlet_Identifiers that doesn't have assigned Oulet_SIze are just three of ten, so in these case we can make a mode imputation.")
st.code("df[['Outlet_Identifier', 'Outlet_Size']].drop_duplicates(['Outlet_Identifier', 'Outlet_Size'])")
st.write(df[['Outlet_Identifier', 'Outlet_Size']].drop_duplicates(['Outlet_Identifier', 'Outlet_Size']))

st.markdown('- The mode for Oulet_Size is "Medium"')
st.write(df[['Item_Identifier','Outlet_Size']].groupby('Outlet_Size').count())
df['Outlet_Size'].fillna('Medium', inplace= True)
st.code("df['Outlet_Size'].fillna('Medium', inplace= True)")

st.markdown('- Results after mode imputation for Outlet_Size')
st.text(df.isnull().sum()*100/df.shape[0])

st.markdown('It is neccesary to search for values that can be an error.')

st.dataframe(df.describe())

st.markdown("""In this case we can observe how the the minimun value for Item_Visibility is zero, if we are talking about items solds, how it is possible to sell something that is not visible?
This should be reviewed with the stakeholders, to establish if it is an error or not.

So we are going to assign the median item visibility for item to the items with zero visibility""")

#Determine average visibility of a product
#visibility_avg = df.pivot_table(values='Item_Visibility', index='Item_Identifier')
visibility_avg= df[df['Item_Visibility']>0][['Item_Identifier','Item_Visibility']].groupby('Item_Identifier').mean()
visibility_avg.reset_index(drop = False, inplace = True)

df = pd.merge(df, visibility_avg, how='left', on = 'Item_Identifier')
df.loc[df['Item_Visibility_x'] == 0, 'Item_Visibility_x'] = df['Item_Visibility_y']
df.drop(columns=['Item_Visibility_y'], inplace=True)
df.rename(columns={'Item_Visibility_x':'Item_Visibility'}, inplace=True)


st.dataframe(df)

st.markdown('Now we need to validate the number of categories for the catergorical attributes and the number of occurrencies, this can be done via visualization')
""""""""""""
vis_data = df[['Item_Identifier','Item_Fat_Content']].groupby('Item_Fat_Content').count()
bar_chart = px.bar(vis_data,
                        x = vis_data.index,
                        y = 'Item_Identifier',
                        text= 'Item_Identifier',
                        template = 'plotly_white',
                        color = vis_data.index,
                        width = 600,
                        height= 300,
                        title = 'Item_Fat_Content values')

bar_chart.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
st.plotly_chart(bar_chart)
st.markdown('''We can see the existence of categories like "Low Fat", "LF" and "low fat" than
can be merge into one called "Low Fat", and the categories "Regular" and "reg" into "Regular"''')

df['Item_Fat_Content'].mask(df['Item_Fat_Content'] == "LF", 'Low Fat', inplace = True)
df['Item_Fat_Content'].mask(df['Item_Fat_Content'] == "low fat", 'Low Fat', inplace = True)
df['Item_Fat_Content'].mask(df['Item_Fat_Content'] == "reg", 'Regular', inplace = True)


vis_data = df[['Item_Identifier','Item_Fat_Content']].groupby('Item_Fat_Content').count()
bar_chart = px.bar(vis_data,
                        x = vis_data.index,
                        y = 'Item_Identifier',
                        text= 'Item_Identifier',
                        template = 'plotly_white',
                        color = vis_data.index,
                        width = 600,
                        height= 300,
                        title = 'Item_Fat_Content values corrected')

bar_chart.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
st.plotly_chart(bar_chart)
""""""""""""

""""""""""""
vis_data2 = df[['Item_Identifier','Item_Type']].groupby('Item_Type').count()
bar_chart2 = px.bar(vis_data2,
                        x = vis_data2.index,
                        y = 'Item_Identifier',
                        text= 'Item_Identifier',
                        template = 'plotly_white',
                        color = vis_data2.index,
                        width = 600,
                        height= 300,
                        title = 'Item_Type values')

bar_chart2.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
st.plotly_chart(bar_chart2)
st.markdown('''We can see the existence of several categories with few items, so we can think in 
create global categories. In this case we can use the Item_Identifier to create three different categories:
- FD: Food
- NC: Non Consumables
- DR: Drinks''')

df['Item_Type'].mask(df['Item_Identifier'].str.startswith('FD')== True, 'Food', inplace = True)
df['Item_Type'].mask(df['Item_Identifier'].str.startswith('NC')== True, 'Non Consumables', inplace = True)
df['Item_Type'].mask(df['Item_Identifier'].str.startswith('DR')== True, 'Drinks', inplace = True)


vis_data3 = df[['Item_Identifier','Item_Type']].groupby('Item_Type').count()
bar_chart3 = px.bar(vis_data3,
                        x = vis_data3.index,
                        y = 'Item_Identifier',
                        text= 'Item_Identifier',
                        template = 'plotly_white',
                        color = vis_data3.index,
                        width = 600,
                        height= 300,
                        title = 'Item_Type values corrected')

bar_chart3.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
st.plotly_chart(bar_chart3)
""""""""""""

""""""""""""
vis_data2 = df[['Item_Identifier','Outlet_Size']].groupby('Outlet_Size').count()
bar_chart2 = px.bar(vis_data2,
                        x = vis_data2.index,
                        y = 'Item_Identifier',
                        text= 'Item_Identifier',
                        template = 'plotly_white',
                        color = vis_data2.index,
                        width = 600,
                        height= 300,
                        title = 'Outlet_Size values')

bar_chart2.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
st.plotly_chart(bar_chart2)

st.markdown('''Nothing wrong with Outlet Size''')

""""""""""""

""""""""""""
vis_data2 = df[['Item_Identifier','Outlet_Location_Type']].groupby('Outlet_Location_Type').count()
bar_chart2 = px.bar(vis_data2,
                        x = vis_data2.index,
                        y = 'Item_Identifier',
                        text= 'Item_Identifier',
                        template = 'plotly_white',
                        color = vis_data2.index,
                        width = 600,
                        height= 300,
                        title = 'Outlet_Location_Type values')

bar_chart2.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
st.plotly_chart(bar_chart2)

st.markdown('''Nothing wrong with Outlet Location Type''')
""""""""""""

""""""""""""
vis_data2 = df[['Item_Identifier','Outlet_Type']].groupby('Outlet_Type').count()
bar_chart2 = px.bar(vis_data2,
                        x = vis_data2.index,
                        y = 'Item_Identifier',
                        text= 'Item_Identifier',
                        template = 'plotly_white',
                        color = vis_data2.index,
                        width = 600,
                        height= 300,
                        title = 'Outlet_Type values')

bar_chart2.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
st.plotly_chart(bar_chart2)

st.markdown('''Nothing wrong with Outlet Type''')
st.markdown('''Now a check to the numeric data, to establish if we need a tranformation or validate some values''')

st.dataframe(df.describe())

st.markdown('''The outlet establisment year has more meaning if we transform the year to the outlet age,
 so we calculate the outlet age up to 2013.''')

df['Outlet_Age'] = 2013 - df['Outlet_Establishment_Year']

temp_cols=df.columns.tolist()
index=df.columns.get_loc("Item_Outlet_Sales")
new_cols= temp_cols[0:index] + temp_cols[index+1:] + temp_cols[index:index+1]
df=df[new_cols]

df.drop(columns=['Outlet_Establishment_Year'], inplace = True)

st.dataframe(df)
""""""""""""
st.markdown('## :blue[8.3. Exploratory Data Analysis]')

st.markdown("""In this EDA we are going to create some visualizations so that you can interact
and look for some outliers, understand the bahaviour of numerical variables vs the independent variable 
and establish if there are correlations between those variables. 
""")

# Histograms
s1 = df.columns.sort_values(ascending = False)
measure_h = st.selectbox('Attribute:', s1)

fig = px.histogram(df, x= measure_h, 
                        template = 'plotly_white',
                        width = 600,
                        height= 500,
                        title = 'Histogram for: ' + measure_h)
st.plotly_chart(fig)

# Boxplots
fig2 = px.box(df, x= measure_h, 
                        template = 'plotly_white',
                        width = 600,
                        height= 500,
                        title = 'Boxplot for: ' + measure_h)
st.plotly_chart(fig2)

# Scatterplots
fig3 = px.scatter(df, x= measure_h, 
                    y = 'Item_Outlet_Sales',
                        template = 'plotly_white',
                        width = 600,
                        height= 500,
                        title = 'Item_Outlet_Sales vs ' + measure_h)
st.plotly_chart(fig3)

# Correlation heatmap
corr_a = df.corr().round(2)
fig4 = px.imshow(corr_a, text_auto = True,
                        template = 'plotly_white',
                        width = 600,
                        height= 500,
                        title = 'Correlation between attributes')
st.plotly_chart(fig4)

st.markdown('## :blue[9. Data Preparation]')

st.markdown("""
We are going to split the data into train and test, this is neccesary to do it before
data normalization and encoding, due to the data leakage problematic. The train data set is used to 
create a model and the test is used to validate the accuracy of that model in non seen data.
The best model can be considered the one that best generalizes the new data. 

- X are the attributes (independent variables)
- y is the dependent variable, the variable we wanto to predict.

""")

from sklearn.model_selection import train_test_split
@st.cache(allow_output_mutation=True)
def split1():

    
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    return X_train, X_test, y_train, y_test, X, y

X_train, X_test, y_train, y_test, X, y = split1()

st.write(f'X_train shape: ', X_train.shape)
st.dataframe(X_train.head())
st.write(f'X_test shape: ', X_test.shape)
st.dataframe(X_test.head())
st.write(f'y_train shape: ', y_train.shape)
st.dataframe(y_train.head())
st.write(f'y_test shape: ', y_test.shape)
st.dataframe(y_test.head())

st.markdown("""
It is neccesary to normalize the data, due our numerical data are in different measures and scales, and the
regression models could missinterpret regressors weights. In this case we are going to use the *MinMaxScaler*.
This is done separately in the X_train and X_test data, in order to prevent data leakage.
""")

from sklearn.preprocessing import MinMaxScaler
@st.cache(allow_output_mutation=True)
def norm1():

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train['Item_Weight'] = scaler.fit_transform(X_train['Item_Weight'].values.reshape(-1,1))
    X_train['Item_Visibility'] = scaler.fit_transform(X_train['Item_Visibility'].values.reshape(-1,1))
    X_train['Item_MRP'] = scaler.fit_transform(X_train['Item_MRP'].values.reshape(-1,1))
    X_train['Outlet_Age'] = scaler.fit_transform(X_train['Outlet_Age'].values.reshape(-1,1))

    X_test['Item_Weight'] = scaler.fit_transform(X_test['Item_Weight'].values.reshape(-1,1))
    X_test['Item_Visibility'] = scaler.fit_transform(X_test['Item_Visibility'].values.reshape(-1,1))
    X_test['Item_MRP'] = scaler.fit_transform(X_test['Item_MRP'].values.reshape(-1,1))
    X_test['Outlet_Age'] = scaler.fit_transform(X_test['Outlet_Age'].values.reshape(-1,1))

    return X_test, X_train, scaler

X_test, X_train, scaler = norm1()

st.markdown("""Then encoding for categorical data is neccesary, the algorithms we will use requiered this process.
We are going to use the *get_dummies* method, this creates additional columns for every categorical value to assign boolean values.
This is done in the X_train and X_test data for the attributes that no are the Item_Identifier and Outlet_Identifier, because this are 
required as identifiers, so our predictor are going to be all the columns except the Item and Outlet identifiers. 
Here you can see the results:
""")

X_test = pd.get_dummies(X_test, columns=['Item_Fat_Content','Item_Type','Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])
X_train = pd.get_dummies(X_train, columns=['Item_Fat_Content','Item_Type','Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])
X_test = X_test.reindex(columns = X_train.columns, fill_value=0)

X_train_pre = X_train.loc[:,~X_train.columns.isin(['Item_Identifier','Outlet_Identifier'])]
X_test_pre = X_test.loc[:,~X_test.columns.isin(['Item_Identifier','Outlet_Identifier'])]


st.write(f'X_train_pre: ', X_train_pre.shape)
st.dataframe(X_train_pre.head())

st.markdown('## :blue[10. Modeling]')

st.markdown('''In the modeling fase we are going to create models with four different supervised 
regression algorithms:
1. Linear regression
2. Ridge regression
3. Lasso regression
4. Polynomial regression 

for everyone of these algorithms we are going to change some parameters in order to find the algorithm
with the lowest Mean Absolute Error (MAE) and the highest r square score. For this we must made a model 
prediction and evaluation. 
''')

st.markdown(':blue[1. Linear regression]')

st.markdown('''So we import the requiered libraries, for the model training, evaluation and selection.
''')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

st.code('''from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score''')

st.markdown('We train the model using the data sets and obtain the predictions')

@st.cache(allow_output_mutation=True)
def linreg1():
    linreg = LinearRegression().fit(X_train_pre, y_train)
    y_predict = linreg.predict(X_test_pre)
    return linreg, y_predict
linreg, y_predict = linreg1()


st.code('''linreg = LinearRegression().fit(X_train_pre, y_train)
y_predict = linreg.predict(X_test_pre)''')

st.markdown('Predictions made for the linear model for the X_test_pre data:')
st.dataframe(y_predict)

st.markdown('Having the model trained and the predictions, we can establish our metrics:')
st.write('- R-squared score (training): {:.3f}'.format(linreg.score(X_train_pre, y_train)))
st.write('- R-squared score (test): {:.3f}'.format(r2_score(y_test, y_predict)))
st.write('- Mean absolute error (linear model): {:.3f}'.format(mean_absolute_error(y_test, y_predict)))

# Store r-squared values, MAD and attributes weights.
@st.cache(allow_output_mutation=True)
def storelin():
    linreg_rs_train = linreg.score(X_train_pre, y_train)
    linreg_rs_test = r2_score(y_test, y_predict)
    linreg_mae = mean_absolute_error(y_test, y_predict)
    linreg_weights = pd.DataFrame(pd.Series(linreg.coef_, X_test_pre.columns), columns=['Linear_coef'])
    return linreg_rs_train, linreg_rs_test, linreg_mae, linreg_weights

linreg_rs_train, linreg_rs_test, linreg_mae, linreg_weights = storelin()

st.markdown('''We can observe how for this model the r squared observed for the train and test data is 
near 0.56 been 1.0 an optimal fit, with a Mean Absolute Error of 859.881.
But this results are achived with the training set initilly cretaed, if we want to see how the r squared 
is afected for k-fold validation (creation of k train and test mini sets with the trainig data), we should use a 
cross validation.''')

st.code('''# Cross validation for linear regression
cv_score_lr = cross_val_score(LinearRegression(), X_train_pre, y_train, cv=20)
print(cv_score_lr.mean())''')

# Cross validation for linear regression
cv_score_lr = cross_val_score(LinearRegression(), X_train_pre, y_train, cv=20)
st.write('The mean r squared for the croos validation is: {:.3f}'.format(cv_score_lr.mean()))

st.markdown('''So if we repeat the training of the model 20 times, taking subsets of data, the median result for the r squared of the models should be 0.558
In the case of lineal regression we don't have parameters to tune or select.''')

st.markdown(':blue[2. Ridge regression]')

from sklearn.linear_model import Ridge
st.code('''# Import the required libreries
from sklearn.linear_model import Ridge''')

st.markdown('''For the Ridge Regression we can tune it with the *alpha* parameter, so we use a Grid Search and 
k-fold in order to iterate over different values for alpha and obtain the value that maximizes the  r square.
And with that value create the model.''')

st.code('''rid_grid = {'alpha':[0.001, 0.01, 0.1, 1, 10, 20, 30, 50, 100, 1000]}
rid_cv = GridSearchCV(estimator = Ridge(), param_grid = rid_grid, cv = 20 ) 
rid_cv.fit(X_train_pre, y_train)
print('The best model is created with: ', rid_cv.best_params_)
''')
@st.cache(allow_output_mutation=True)
def gridlin():
    rid_grid = {'alpha':[0.001, 0.01, 0.1, 1, 10, 20, 30, 50, 100, 1000]}
    rid_cv = GridSearchCV(estimator = Ridge(), param_grid = rid_grid, cv = 20 ) 
    rid_cv.fit(X_train_pre, y_train)
    return rid_grid, rid_cv

rid_grid, rid_cv = gridlin()

st.write('The best model is created with: ', rid_cv.best_params_)

st.markdown('Now knowing the best value parameter to use we train the model using the data sets and obtain the predictions.')

@st.cache(allow_output_mutation=True)
def ridgereg1():
    ridreg = Ridge(alpha = 0.1).fit(X_train_pre, y_train)
    y_predict = ridreg.predict(X_test_pre)
    return ridreg, y_predict

ridreg, y_predict = ridgereg1()

st.code('''ridreg = Ridge(alpha = 0.1).fit(X_train_pre, y_train)
y_predict = ridreg.predict(X_test_pre)''')

st.markdown('Predictions made for the Ridge Regression model for the X_test_pre data:')
st.dataframe(y_predict)

st.markdown('Having the model trained and the predictions, we can establish our metrics:')
st.write('- R-squared score (training): {:.3f}'.format(ridreg.score(X_train_pre, y_train)))
st.write('- R-squared score (test): {:.3f}'.format(r2_score(y_test, y_predict)))
st.write('- Mean absolute error (Ridge model): {:.3f}'.format(mean_absolute_error(y_test, y_predict)))

st.markdown('So our models have a similar result, or at least the r squared and MAE are.')

# Store r-squared values, MAD and attributes weights.

@st.cache(allow_output_mutation=True)
def storerid():
    ridreg_rs_train = ridreg.score(X_train_pre, y_train)
    ridreg_rs_test = r2_score(y_test, y_predict)
    ridreg_mae = mean_absolute_error(y_test, y_predict)
    ridreg_weights = pd.DataFrame(pd.Series(ridreg.coef_, X_test_pre.columns), columns=['Ridge_coef'])
    return ridreg_rs_train, ridreg_rs_test, ridreg_mae, ridreg_weights

ridreg_rs_train, ridreg_rs_test, ridreg_mae, ridreg_weights = storerid()

st.markdown(':blue[3. Lasso regression]')

from sklearn.linear_model import Lasso
st.code('''# Import the required libreries
from sklearn.linear_model import Lasso''')

st.markdown('''For the Lasso Regression we can tune it with the *alpha* parameter, so we use a Grid Search and 
k-fold in order to iterate over different values for alpha and obtain the value that maximizes the r square.
And with that value create the model.''')

st.code('''lasso_grid = {'alpha':[0.001, 0.01, 0.1, 1, 10, 20, 30, 50, 100, 1000]}
lasso_cv = GridSearchCV(estimator = Lasso(), param_grid = lasso_grid, cv = 20 ) 
lasso_cv.fit(X_train_pre, y_train)
print('The best model is created with: ', lasso_cv.best_params_)
''')

@st.cache(allow_output_mutation=True)
def lassoreg1():
    lasso_grid = {'alpha':[0.001, 0.01, 0.1, 1, 10, 20, 30, 50, 100, 1000]}
    lasso_cv = GridSearchCV(estimator = Lasso(), param_grid = lasso_grid, cv = 20 ) 
    lasso_cv.fit(X_train_pre, y_train)
    return lasso_grid, lasso_cv

lasso_grid, lasso_cv = lassoreg1()

st.write('The best model is created with: ', lasso_cv.best_params_)

st.markdown('Now knowing the best value parameter to use we train the model using the data sets and obtain the predictions.')
lassoreg = Lasso(alpha = 0.1).fit(X_train_pre, y_train)
y_predict = lassoreg.predict(X_test_pre)

st.code('''lassoreg = Lasso(alpha = 0.001).fit(X_train_pre, y_train)
y_predict = lassoreg.predict(X_test_pre)''')

st.markdown('Predictions made for the Lasso Regression model for the X_test_pre data:')
st.dataframe(y_predict)

st.markdown('Having the model trained and the predictions, we can establish our metrics:')
st.write('- R-squared score (training): {:.3f}'.format(lassoreg.score(X_train_pre, y_train)))
st.write('- R-squared score (test): {:.3f}'.format(r2_score(y_test, y_predict)))
st.write('- Mean absolute error (Lasso model): {:.3f}'.format(mean_absolute_error(y_test, y_predict)))

st.markdown('So our models have a similar result, or at least the r squared and MAE are.')

# Store r-squared values, MAD and attributes weights.

@st.cache(allow_output_mutation=True)
def storelasso():
    lassoreg_rs_train = lassoreg.score(X_train_pre, y_train)
    lassoreg_rs_test = r2_score(y_test, y_predict)
    lassoreg_mae = mean_absolute_error(y_test, y_predict)
    lassoreg_weights = pd.DataFrame(pd.Series(lassoreg.coef_, X_test_pre.columns), columns=['Lasso_coef'])
    return lassoreg_rs_train, lassoreg_rs_test, lassoreg_mae, lassoreg_weights

lassoreg_rs_train, lassoreg_rs_test, lassoreg_mae, lassoreg_weights = storelasso()

st.markdown(':blue[4. Polynomial regression]')

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

st.code('''# Import the required libreries
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures''')

st.markdown('''For the Polynomial regression we need to transform our features into polynomial features,
in this case we are going to work with quadratic features or degree 2. And because the addiction of polynomial features 
in combination with regression leads to overfitting, we are going to use a Ridge regression that has
a regularization penalty (alpha parameter)''')

st.write('\nNow we transform the original input data to add\n\
polynomial features up to degree 2 (quadratic)\n')

st.code('''poly = PolynomialFeatures(degree=2)
X_F2_poly = poly.fit_transform(X[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']])
X_F2_poly = pd.DataFrame(X_F2_poly)
X_F2_poly = pd.concat([X_F2_poly, X[['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_F2_poly, y, random_state = 0)
''')

@st.cache(allow_output_mutation=True)
def polytrans():
    poly = PolynomialFeatures(degree=2)
    X_F2_poly = poly.fit_transform(X[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']])
    X_F2_poly = pd.DataFrame(X_F2_poly)
    X_F2_poly = pd.concat([X_F2_poly, X[['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']]], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_F2_poly, y, random_state = 0)
    return poly, X_F2_poly, X_train, X_test, y_train, y_test

poly, X_F2_poly, X_train, X_test, y_train, y_test = polytrans()

st.markdown('X_train data:')

st.markdown('''It is neccesary to transform the data again, it means to normalize and create dummy variables for 
the categorical features.
''')

@st.cache(allow_output_mutation=True)
def scaled():

    scaler = MinMaxScaler()

    i = 1
    while i != 15:
        X_train.iloc[:,i] = scaler.fit_transform(X_train.iloc[:,i].values.reshape(-1,1))
        i +=1

    j = 1
    while j != 15:
        X_test.iloc[:,j] = scaler.fit_transform(X_test.iloc[:,j].values.reshape(-1,1))
        j +=1
    return X_train, X_test

X_train, X_test = scaled()

st.dataframe(X_test.head())


X_test = pd.get_dummies(X_test, columns=['Item_Fat_Content','Item_Type','Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])
X_train = pd.get_dummies(X_train, columns=['Item_Fat_Content','Item_Type','Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])
X_test = X_test.reindex(columns = X_train.columns, fill_value=0)

@st.cache(allow_output_mutation=True)
def poly_t1():
    X_train_pre = X_train.loc[:,~X_train.columns.isin(['Item_Identifier','Outlet_Identifier'])]
    X_test_pre = X_test.loc[:,~X_test.columns.isin(['Item_Identifier','Outlet_Identifier'])]
    return X_train_pre, X_test_pre

X_train_pre, X_test_pre = poly_t1()

st.write(f'X_train_pre: ', X_train_pre.shape)
st.dataframe(X_train_pre.head())


st.code('''poly_lasso_grid = {'alpha':[0.001, 0.01, 0.1, 1, 10, 20, 30, 50, 100, 1000]}
poly_lasso_cv = GridSearchCV(estimator = Lasso(), param_grid = poly_lasso_grid, cv = 20 ) 
poly_lasso_cv.fit(X_train_pre, y_train)
print('The best model is created with: ', poly_lasso_cv.best_params_)
''')

@st.cache(allow_output_mutation=True)
def poly_fit():
    poly_lasso_grid = {'alpha':[0.001, 0.01, 0.1, 1, 10, 20, 30, 50, 100, 1000]}
    poly_lasso_cv = GridSearchCV(estimator = Lasso(), param_grid = poly_lasso_grid, cv = 20 ) 
    poly_lasso_cv.fit(X_train_pre, y_train)
    return poly_lasso_cv

poly_lasso_cv = poly_fit()

st.write('The best model is created with: ', poly_lasso_cv.best_params_)

st.markdown('Now knowing the best value parameter to use we train the model using the data sets and obtain the predictions.')
poly_lassoreg = Lasso(alpha = 0.1).fit(X_train_pre, y_train)
y_predict = poly_lassoreg.predict(X_test_pre)

st.code('''poly_lassoreg = Lasso(alpha = 0.1).fit(X_train_pre, y_train)
y_predict = poly_lassoreg.predict(X_test_pre)''')

st.markdown('Predictions made for the Polynomial Lasso Regression model for the X_test_pre data:')
st.dataframe(y_predict)

st.markdown('Having the model trained and the predictions, we can establish our metrics:')
st.write('- R-squared score (training): {:.3f}'.format(poly_lassoreg.score(X_train_pre, y_train)))
st.write('- R-squared score (test): {:.3f}'.format(r2_score(y_test, y_predict)))
st.write('- Mean absolute error (Polynomial Lasso model): {:.3f}'.format(mean_absolute_error(y_test, y_predict)))

st.markdown('So our models have a similar result, or at least the r squared and MAE are.')

# Store r-squared values, MAD and attributes weights.

@st.cache(allow_output_mutation=True)
def store_poly():
    poly_lassoreg_rs_train = poly_lassoreg.score(X_train_pre, y_train)
    poly_lassoreg_rs_test = r2_score(y_test, y_predict)
    poly_lassoreg_mae = mean_absolute_error(y_test, y_predict)
    poly_lassoreg_weights = pd.DataFrame(pd.Series(poly_lassoreg.coef_, X_test_pre.columns))
    return poly_lassoreg_rs_train, poly_lassoreg_rs_test, poly_lassoreg_mae, poly_lassoreg_weights

poly_lassoreg_rs_train, poly_lassoreg_rs_test, poly_lassoreg_mae, poly_lassoreg_weights = store_poly()


st.markdown('### :blue[Model selection]')
st.markdown('''We have trained four different regression models, so it is time to evaluate them and
select the one with the best results.

Here we have the metrics of our models:''')

st.write(f'''
| Metric | 1. Linear regression | 2. Lasso regression| 3. Ridge regression| 4. Polynomial ridge regression|
| ----------- | ----------- | ----------- | ----------- | ----------- |
| R square train data |{linreg_rs_train:.3f}|{lassoreg_rs_train:.3f}|{ridreg_rs_train:.3f}|{poly_lassoreg_rs_train:.3f}|
| R square test data |{linreg_rs_test:.3f}|{lassoreg_rs_test:.3f}|{ridreg_rs_test:.3f}|{poly_lassoreg_rs_test:.3f}|
| Mean absolute error |{linreg_mae:.3f}|{lassoreg_mae:.3f}|{ridreg_mae:.3f}|{poly_lassoreg_mae:.3f}|

''')

st.markdown('''Based in our metrics we can select the *polynomial ridge regression* as the best model,
due this one has the maximum r square for the train and test data, and has the minimum mean absolute error.

Here we can compare the feature weights for each model that not is the polynomial:''')

w_res_df = linreg_weights.merge(ridreg_weights, how = 'inner', left_index=True, right_index=True)
w_res_df = w_res_df.merge(lassoreg_weights, how = 'inner', left_index=True, right_index=True)
st.dataframe(w_res_df)

st.markdown('''We can see how the weights of the attributes or independent variables are almost entirely high for the linear regression, 
while for the ridge and lasso the values decrease considerably.
In addition, there are some attributes that could be eliminated from the model, since their weights are lower than others, 
and seem to contribute very little to the target variable (sales). It should be recalled that these weights 
are computed against values ranging from 0 to 1, given the data transformations used.

Models could be built in which variables such as Item_Weight, Item_Visibility, 
Item_Fat_Content and Item_Type are eliminated, to determine if better fits are achieved. As well as the use of polynomial regressions 
with higher degrees than the quadratic one used here, or the inclusion of other machime learning algorithms such as
decision trees, Random Forest or running algorithms such as GBM and XGBoost.

In this case we are going to keep the polynomial regression model obtained, which has the following weights: 
 ''')

poly_lassoreg_weights = poly_lassoreg_weights.rename(columns = {0:'Poly regres coef'})
st.dataframe(poly_lassoreg_weights)


st.markdown('## :blue[11. Deployment]')

st.markdown('''By way of deployment, an interface is created in which the user can select values for the different attributes and receive a sales 
forecast according to the model built with the second degree polynomial regression.''')

lst_Item_Fat_Content = X['Item_Fat_Content'].unique()
lst_Item_Type = X['Item_Type'].unique()
lst_Outlet_Size = X['Outlet_Size'].unique()
lst_Outlet_Location_Type = X['Outlet_Location_Type'].unique()
lst_Outlet_Type = X['Outlet_Type'].unique()

c1, c2, c3, c4 = st.columns(4)

with c1:
    u_Item_Weight = st.slider('Item_Weight', min_value = 1.0, max_value = 25.0, step = 0.1)

with c2:
    u_Item_Visibility = st.slider('Item_Visibility', min_value = 0.0, max_value = 1.0, step = 0.1)

with c3:
    u_Item_MRP = st.slider('Item_MRP', min_value = 1.0, max_value = 3000.0, step = 0.1)

with c4:
    u_Outlet_Age = st.slider('Outlet_Age', min_value = 1.0, max_value = 100.0, step = 0.1)

z1, z2, z3 = st.columns(3)

with z1:
    u_Item_Fat_Content = st.selectbox('Item_Fat_Content:', lst_Item_Fat_Content)

with z2:
    u_Item_Type  = st.selectbox('Item_Type:', lst_Item_Type)

with z3:
    u_Outlet_Size = st.selectbox('Outlet_Size:', lst_Outlet_Size)


z11, z22= st.columns(2)

with z11:
    u_Outlet_Location_Type = st.selectbox('Outlet_Location_Type:', lst_Outlet_Location_Type)

with z22:
    u_Outlet_Type = st.selectbox('Outlet_Type:', lst_Outlet_Type)




@st.cache(allow_output_mutation=True)
def final():
    MinMax_df = pd.DataFrame(X_F2_poly.describe().loc[['min','max']])

    u_nume = (u_Item_Weight, u_Item_Visibility, u_Item_MRP, u_Outlet_Age)

    u_poly = PolynomialFeatures(degree=2)
    u_X_F2_poly = pd.DataFrame(u_poly.fit_transform(np.array(u_nume).reshape(1, -1)))

    for i in u_X_F2_poly.columns[1:]:
        u_X_F2_poly.iloc[:,i] = (u_X_F2_poly.iloc[:,i] - MinMax_df.loc['min',i])/ (MinMax_df.loc['max',i] - MinMax_df.loc['min',i])

    u_cate = pd.DataFrame(data= (np.array([u_Item_Fat_Content, u_Item_Type, u_Outlet_Size, u_Outlet_Location_Type, u_Outlet_Type]).reshape(1,-1)), columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])
    u_X_F2_poly = pd.concat([u_X_F2_poly, u_cate], axis = 1)
    df_pred = pd.get_dummies(u_X_F2_poly, columns=['Item_Fat_Content','Item_Type','Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])
    df_pred_u = pd.DataFrame(data = np.zeros(30).reshape(1,-1) ,columns = X_test_pre.columns)

    for i in df_pred.columns:
        df_pred_u[i] = df_pred[i]

    sales_predi = poly_lassoreg.predict(df_pred_u)

    return sales_predi

sales_predi = final()

st.write('The predicted sale is: ', sales_predi)


