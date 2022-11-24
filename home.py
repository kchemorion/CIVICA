import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from streamlit_modal import Modal
import re
import plotly.figure_factory as ff
import time

st.header('ENVSAVE')
st.write('Forest Fires Prediction and Hyper-Parameter Experimentation Tool')

modal = Modal("ENVSAVE Data Schema", key='modal1')
open_modal = st.button("Click to check the required data schema")
if open_modal:
    modal.open()

if modal.is_open():
    with modal.container():
        st.write('X : x-axis spatial coordinate within the Montesinho park map: 1 to 9')
        st.write('Y : y-axis spatial coordinate within the Montesinho park map: 2 to 9')
        st.write('month : month of the year: Jan to Dec')
        st.write('day : day of the week: Mon to Sun')
        st.write('FFMC : FFMC (Fine Fuel Moisture Code) index from the FWI system: 18.7 to 96.20')
        st.write('DMC : DMC (Duff Moisture Code) index from the FWI system: 1.1 to 291.3')
        st.write('DC : DC (Drought Code) index from the FWI system: 7.9 to 860.6 ')
        st.write('ISI : ISI (Initial Spread Index) index from the FWI system: 0.0 to 56.10 ')
        st.write('temp : temperature in Celsius degrees: 2.2 to 33.30')
        st.write('RH : relative humidity in %: 15.0 to 100' )
        st.write('wind : wind speed in km/h: 0.40 to 9.40')
        st.write('rain : outside rain in mm/m2 : 0.0 to 6.4')
        st.write('area : the burned area of the forest (in ha): 0.00 to 1090.84')
dataframe = pd.DataFrame()
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)

    # Add size category to solve the classification problem <6 is 0 and >6 is 1
    dataframe['size_category'] = np.where(dataframe['area']>6, '1', '0')
    dataframe['size_category']= pd.to_numeric(dataframe['size_category'])

    # preprocessing for days  into week and weekends
    dataframe['day'] = ((dataframe['day'] == 'sun') | (dataframe['day'] == 'sat'))
    # renaming column
    dataframe = dataframe.rename(columns = {'day' : 'is_weekend'})

    
if not dataframe.empty:
    tab1, tab2, tab3, tab4 = st.tabs(['Data Preprocessing', 'Distributions', 'Train, Test, Split', 'Experiments'])

    with tab1:
        st.write('Sample data:')
        st.write(dataframe.head())
        st.write('Shape of the data:')
        st.write(dataframe.describe().transpose())

        ### exploring months ###


        ### months to seasons ###

        # dictionary of months mapping to seasons
        season_dict = {'dec' : 'winter', 'jan' : 'winter', 'feb' : 'winter',
                    'mar' : 'spring', 'apr' : 'spring', 'may' : 'spring',
                    'jun' : 'summer', 'jul' : 'summer', 'aug' : 'summer', 'sep' : 'summer',
                    'oct' : 'autumn', 'nov' : 'autumn'}

        # applying dictionary
        dataframe = dataframe.replace({'month' : season_dict})

        # renaming column
        dataframe = dataframe.rename(columns = {'month' : 'season'})

 
        ### converting season to summer or not summer ###

        # converting to is summer
        dataframe['season'] = (dataframe['season'] == 'summer')

        # renaming column
        dataframe = dataframe.rename(columns = {'season' : 'is_summer'})

        # visualizing data
        sns.countplot(dataframe['is_summer'])
        plt.title('Count plot of summer vs other seasons')

        ### exploring days ###

        # visualizing days


        ### turning days into is_weekend ###

        # converting to is weekend
        #dataframe['day'] = ((dataframe['day'] == 'sun') | (dataframe['day'] == 'sat'))

        # renaming column
        dataframe = dataframe.rename(columns = {'day' : 'is_weekend'})

        # visualizing
        sns.countplot(dataframe['is_weekend'])
        plt.title('Count plot of weekend vs weekday')




    with tab2:
        ### visualizing distributions ###

        # subplots
        fig, ax = plt.subplots(11, figsize = (10,50))

        # initial index
        index = 0
        group_labels = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']

        col1, col2 = st.columns(2)
        with col2:
            # visualizing
            for column in group_labels:
                st.write('Distribution of '+ column +'  data')
                np_model = dataframe[column].to_numpy()
                np_model = np.random.normal(1, 1, size=50)
                fig, ax = plt.subplots()
                ax.hist(np_model, bins=20)
                st.pyplot(fig)
        with col1:
            st.write('Correlation plot for '+column+'  data')
            fig, ax = plt.subplots()
            sns.heatmap(dataframe.corr(), ax=ax)
            st.write(fig)

    with tab3:
        # ### train test split ###

        # # separating features from target
        # # normally we use X and y but since X and Y are variables, we use the names features and targets to prevent potential conflicts
        # features = dataframe.drop(['area'], axis = 1)
        # target = dataframe['area'].values.reshape(-1, 1)

        # # splitting into train test set
        # features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.5, random_state = 42)

        # ### scaling data ###

        # # fitting scaler
        # sc_feature = StandardScaler()
        # sc_target = StandardScaler()

        # # transforming features
        # features_test = sc_feature.fit_transform(features_test)
        # features_train = sc_feature.transform(features_train)

        # # transforming target
        # target_test = sc_target.fit_transform(target_test)
        # target_train = sc_target.transform(target_train)

        # ### converting everything to dataframe for csv storage ###

        # # features
        # features_test = pd.DataFrame(features_test, columns = features.columns)
        # features_train = pd.DataFrame(features_train, columns = features.columns)

        # # target
        # target_test = pd.DataFrame(target_test, columns = ['area'])
        # target_train = pd.DataFrame(target_train, columns = ['area'])

        # # checking to see if everything is in order
        # st.write('Test Features Sample')
        # st.write(features_test.head())
        # st.write('Train Features Sample')
        # st.write(features_train.head())

        # if st.button('Split and Save Data'):

        #     ### saving everything to csv ###
        #     # features
        #     features_test.to_csv('data/features_test.csv')
        #     features_train.to_csv('data/features_train.csv')

        #     # target
        #     target_test.to_csv('data/target_test.csv')
        #     target_train.to_csv('data/target_train.csv')

        #     with st.spinner('Wait for it...'):
        #         time.sleep(5)
        #     st.success('Done!')
        features = dataframe.drop(['size_category'], axis = 1)
        labels = dataframe['size_category'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size = 0.2, random_state = 42)

        # fitting scaler
        sc_features = StandardScaler()
        # transforming features
        X_test = sc_features.fit_transform(X_test)
        X_train = sc_features.transform(X_train)
        # features
        X_test = pd.DataFrame(X_test, columns = features.columns)
        X_train = pd.DataFrame(X_train, columns = features.columns)
        # labels
        y_test = pd.DataFrame(y_test, columns = ['size_category'])
        y_train = pd.DataFrame(y_train, columns = ['size_category'])
        st.write(X_train.head())


    

    with tab4:

        nepochs = st.slider('Number of epochs', 0, 250, 25)

        b_size = st.slider('Batch size', 0, 100, 5)

        functions = st.multiselect('Activation Function',
        ['Relu', 'sigmoid'])

        #st.write('You selected:', functions[0])
        

        if st.button('Run Experiment'):
            model = Sequential()
            # input layer + 1st hidden layer
            model.add(Dense(6, input_dim=13, activation='relu'))
            # 2nd hidden layer
            model.add(Dense(6, activation='relu'))
            # output layer
            model.add(Dense(6, activation='sigmoid'))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation = 'relu'))
            model.summary()



            # Compile Model
            model.compile(optimizer = 'adam', metrics=['accuracy'], loss ='binary_crossentropy')
            # Train Model
            history = model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = b_size, epochs = nepochs)    
            _, train_acc = model.evaluate(X_train, y_train, verbose=0)
            _, valid_acc = model.evaluate(X_test, y_test, verbose=0)
            
            with st.spinner('Wait for it...'):
                time.sleep(5)
            st.success('Done!')

            st.write('Train: %.3f, Valid: %.3f' % (train_acc, valid_acc))   

            plt.figure(figsize=[8,5])
            plt.plot(history.history['accuracy'], label='Train')
            plt.plot(history.history['val_accuracy'], label='Valid')
            plt.legend()
            plt.xlabel('Epochs', fontsize=16)
            plt.ylabel('Accuracy', fontsize=16)
            plt.title('Accuracy Curves Epoch 100, Batch Size 10', fontsize=16)
            plt.show()






