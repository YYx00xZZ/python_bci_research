import pickle
import requests
from scipy.sparse import data
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import plotly.graph_objects as go
#knn imports
from sklearn.neighbors import KNeighborsClassifier
from streamlit.elements.arrow import Data
from streamlit.proto.Empty_pb2 import Empty
# from utils import populate_NaN

def main():
	#---------------------------------#
	# Page layout
	## Page expands to full width
	st.set_page_config(page_title='The Machine Learning App',
		layout='wide')

	default_columns = ["AF3","F7","F3","F5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4","EventId"]

	def populate_NaN(dataframe):
		""" за всички NaN полета задаваме стойност 0 """

		dataframe = dataframe.fillna(0)
		return dataframe


	def populate_ids(dataframe):
		""" loop1. Замества ID=0 със съответното правилно """
		
		# dataframe = dataframe.astype({'EventId': 'int32'}).dtypes
		EventID_NEW = ''
		EventID_TO_SET = ''
		for index, row in dataframe.iterrows(): #итерираме ред по ред с вградения метод iterrows() от pandas
			EventID_NEW = str(row['EventId']) # достъпваме данни чрез име на колона
			if EventID_NEW == '0':
				dataframe.at[index, 'EventId'] = str(EventID_TO_SET)
			else:
				EventID_NEW = str(row['EventId'])
				EventID_TO_SET = str(row['EventId'])
		return dataframe
	def filter_events(dataframe):
		""" loop2.
		Изтрива всички редове които не са с id = 33025 или 33026
		Функцията (generator) очаква pandas.DataFrame;
		Връща (generator) DataFrame съдържащ данни само за ляво и дясно
		"""
		dataframe = dataframe[(dataframe.EventId == '33025') | (dataframe.EventId == '33026')]
		return dataframe.astype({'EventId': 'int'})


	def neutral_events(dataframe):
		EventID_NEW = ''
		EventID_TO_SET = ''
		for index, row in dataframe.iterrows(): #итерираме ред по ред с вградения метод iterrows() от pandas
			EventID_NEW = str(row['EventId']) # достъпваме данни чрез име на колона
			# print(EventID_NEW)
			if (EventID_NEW == '33025') or (EventID_NEW == '33026'):
				# dataframe.at[index, 'EventId'] = str('33027')  #(EventID_TO_SET)
				pass
			else:
				dataframe.at[index,'EventId'] = str('33027')
				# EventID_NEW = str('33027') # (row['EventId'])
				# print(f'from else: new {EventID_NEW}')
				# EventID_TO_SET = str('33027') # (row['EventId'])
				# print(f'from else: to set {EventID_TO_SET}')
		return dataframe # .astype({'EventId': 'int'}) # .astype({'EventId': 'int'})


	def equalize_events(dataframe):
		events_count = dataframe.EventId.value_counts()  # или dataframe['EventId'].value_counts(),
														# (почти никога) няма значение

		# type(events_count) резултата е : <class 'pandas.core.series.Series'>
		# st.write(events_count)
		st.write(f"\nТрябва ни размер на матрицата не по-малък, и не по-голям от {events_count.min()} реда.")

		count_class_33024, count_class_33025, count_class_33026 = dataframe.EventId.value_counts()

		# Divide by class
		df_class_33024 = dataframe[dataframe['EventId'] == 33024]
		df_class_33025 = dataframe[dataframe['EventId'] == 33025]
		df_class_33026 = dataframe[dataframe['EventId'] == 33026]
		# print(df_class_33024.shape,df_class_33025.shape,df_class_33026.shape,sep="\t")
		# dataframe.loc[dataframe['EventId'] == 33025]
		# st.write(dataframe.iloc[435:445]) # Ако погледнем към колоната с ивентите, виждаме къде точно започва следващия
		# st.write(df_class_33024)

		df_class_33024_undersample = df_class_33024.sample(count_class_33025)
		df_alex_under = pd.concat([df_class_33024_undersample, df_class_33025,df_class_33026], axis=0) # това е тест сет 

		# st.write('Random under-sampling:')
		# st.write(df_alex_under.EventId.value_counts())

		return df_alex_under


	#---------------------------------#
	# def update_event_ids_all_event(data):
	#     # Генерипа последователно ID за всеки event - може да се използва за TIME серии и др.
	#     chk_33025 = 0
	#     chk_33026 = 0
	#     chk_33027 = 0
	#     chk_33028 = 0
	#     for index, row in data.iterrows(): #итерираме ред по ред с вградения метод iterrows() от pandas
	#         if row['EventId'] == 33025:  
	#             chk_33025 += 1
	#             data.at[index, 'ID_all_events'] = chk_33025
	#         if row['EventId'] == 33026:  
	#             chk_33026 += 1
	#             data.at[index, 'ID_all_events'] = chk_33026
	#         if row['EventId'] == 33027:  
	#             chk_33027 += 1
	#             data.at[index, 'ID_all_events'] = chk_33027
	#         if row['EventId'] == 33028:  
	#             chk_33028 += 1
	#             data.at[index, 'ID_all_events'] = chk_33028            
	#     return data

	#---------------------------------#
	# Model building
	def build_forest_model(df):
		X = df.iloc[:,:-1] # Using all column except for the last column as X
		Y = df.iloc[:,-1] # Selecting the last column as Y

		st.markdown('**1.2. Data splits**')
		st.write('Training set')
		st.info(X.shape)
		st.write('Test set')
		st.info(Y.shape)

		st.markdown('**1.3. Variable details**:')
		st.write('X variable')
		st.info(list(X.columns))
		st.write('Y variable')
		st.info(Y.name)

		# Data splitting
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, stratify=Y)

		rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
			max_depth=parameter_max_depth,
			random_state=parameter_random_state,
			max_features=parameter_max_features,
			criterion=parameter_criterion,
			min_samples_split=parameter_min_samples_split,
			min_samples_leaf=parameter_min_samples_leaf,
			bootstrap=parameter_bootstrap,
			oob_score=parameter_oob_score,
			n_jobs=parameter_n_jobs)
		rf.fit(X_train, Y_train)
		# knc KNeighborsClassifier
		knc = KNeighborsClassifier(n_neighbors=5,
			# *,
			weights='uniform',
			algorithm='auto',
			leaf_size=30,
			p=2,
			metric='minkowski',
			metric_params=None,
			n_jobs=None)
		knc.fit(X_train, Y_train)
		#     #save model
		with open("randomforest_model.pickle", "wb") as file:
			pickle.dump(rf, file)

		st.subheader('2. Model Performance')

		st.markdown('**2.1. Training set**')
		Y_pred_train = rf.predict(X_train)
		st.write('Coefficient of determination ($R^2$):')
		st.info( r2_score(Y_train, Y_pred_train) )

		st.write('Error (MSE or MAE):')
		st.info( mean_squared_error(Y_train, Y_pred_train) )

		st.markdown('**2.2. Test set**')
		Y_pred_test = rf.predict(X_test)
		st.write('Coefficient of determination ($R^2$):')
		st.info( r2_score(Y_test, Y_pred_test) )

		st.write('Error (MSE or MAE):')
		st.info( mean_squared_error(Y_test, Y_pred_test) )

		st.subheader('3. Model Parameters')
		st.write(rf.get_params())
		#open model and use it
		loaded_forest_model = pickle.load(open("randomforest_model.pickle", "rb"))
		# acc = loaded_mlp_model.score(X,Y)
		# st.write(acc)
		# predictions = loaded_mlp_model.predict(X[1,:].reshape(1,-1))
		# st.write(X[1,:])

		# st.write(predictions)
		# st.write(Y[1])
		import re
		# collect_numbers = lambda x : [float(i) for i in re.split("[^0-9]", x) if i != ""]
		collect_numbers = lambda x : [float(i) for i in re.split(",", x) if i != ""]
		numbers = st.text_input("PLease enter 14 values (separated with ,). THis represent 1 row of BCI data for which we`ll predict if it's left or right.")
		st.text('for e.g. #4081.0256, 4091.6667, 4069.6155, 4088.4614, 4099.1025, 4092.3076, 4078.7180, 4078.2051, 4066.7949, 4066.4102, 4046.7949, 4050, 4092.9487, 4081.2820')
		if not numbers:
			st.info("Waiting for input")
		else:
			predictions = loaded_forest_model.predict(np.array(collect_numbers(numbers)).reshape(1,-1))
			st.write('prediction result', predictions)
	def build_knn_model(df):
		X = df.iloc[:,:-1] # Using all column except for the last column as X
		Y = df.iloc[:,-1] # Selecting the last column as Y

		st.markdown('**1.2. Data splits**')
		st.write('Training set')
		st.info(X.shape)
		st.write('Test set')
		st.info(Y.shape)

		st.markdown('**1.3. Variable details**:')
		st.write('X variable')
		st.info(list(X.columns))
		st.write('Y variable')
		st.info(Y.name)

		# Data splitting
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, stratify=Y)

		# rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
		#     max_depth=parameter_max_depth,
		#     random_state=parameter_random_state,
		#     max_features=parameter_max_features,
		#     criterion=parameter_criterion,
		#     min_samples_split=parameter_min_samples_split,
		#     min_samples_leaf=parameter_min_samples_leaf,
		#     bootstrap=parameter_bootstrap,
		#     oob_score=parameter_oob_score,
		#     n_jobs=parameter_n_jobs)
		# rf.fit(X_train, Y_train)
		# knc KNeighborsClassifier
		knc = KNeighborsClassifier(n_neighbors=parameter_n_neighbors,
			weights=parameter_weights,
			algorithm=parameter_algorithm,
			leaf_size=parameter_leaf_size,
			n_jobs=parameter_n_jobs)
		knc.fit(X_train, Y_train)
		#     #save model
		with open("knn_model.pickle", "wb") as file:
			pickle.dump(knc, file)

		st.subheader('2. Model Performance')

		st.markdown('**2.1. Training set**')
		Y_pred_train = knc.predict(X_train)
		st.write('Coefficient of determination ($R^2$):')
		st.info( r2_score(Y_train, Y_pred_train) )

		st.write('Error (MSE or MAE):')
		st.info( mean_squared_error(Y_train, Y_pred_train) )

		st.markdown('**2.2. Test set**')
		Y_pred_test = knc.predict(X_test)
		st.write('Coefficient of determination ($R^2$):')
		st.info( r2_score(Y_test, Y_pred_test) )

		st.write('Error (MSE or MAE):')
		st.info( mean_squared_error(Y_test, Y_pred_test) )

		st.subheader('3. Model Parameters')
		st.write(knc.get_params())
		#open model and use it
		loaded_knc_model = pickle.load(open("knn_model.pickle", "rb"))
		# predict your data here with the loaded model
		# import re
		# # collect_numbers = lambda x : [float(i) for i in re.split("[^0-9]", x) if i != ""]
		# collect_numbers = lambda x : [float(i) for i in re.split(",", x) if i != ""]
		# to_predict=[]
		# numbers = st.text_input("PLease enter 14 values (separated with ,). THis represent 1 row of BCI data for which we`ll predict if it's left or right.")
		# st.text('for e.g. #4081.0256, 4091.6667, 4069.6155, 4088.4614, 4099.1025, 4092.3076, 4078.7180, 4078.2051, 4066.7949, 4066.4102, 4046.7949, 4050, 4092.9487, 4081.2820')
		# if not numbers:
		# 	st.info("Waiting for input")
		# else:
		# 	to_predict.append(collect_numbers(numbers))
		# 	predictions = loaded_knc_model.predict(np.array(collect_numbers(numbers)).reshape(1,-1))
		# 	st.write('prediction result', predictions)


	#---------------------------------#
	st.write("""
	# ml app
	""")

	st.code("a da vidim")


	#---------------------------------#
	# Sidebar - Collects user input features into dataframe
	with st.sidebar.header('1. Upload your CSV data'):
		# Select alg
		classifier = st.sidebar.selectbox("Select model", ("---", "Random Forest", "KNN"))
		print(classifier)
		uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv", "xlsx"])
		with st.sidebar.header('2. Set Parameters'):
			split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)


	if classifier == "Random Forest":
		# Sidebar - Specify parameter settings

		with st.sidebar.subheader('2.1. Learning Parameters'):
			parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
			parameter_max_depth = st.sidebar.slider('Max depth', 0,50,20,5)
			parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
			parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
			parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

		with st.sidebar.subheader('2.2. General Parameters'):
			parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
			parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
			parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
			parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
			parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])
	if classifier == "KNN":
		with st.sidebar.subheader('2.1. Parameters'):
			parameter_n_neighbors = st.sidebar.slider('n_neighbors', 0,20,5,1)
			parameter_weights = st.sidebar.select_slider('weights', options=['uniform', 'distance'])
			parameter_algorithm = st.sidebar.selectbox('Select algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
			parameter_leaf_size = st.sidebar.slider('leaf_size', 0,120,30,5)
			# parameter_p ???
			parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])
	#---------------------------------#

	# Main panel
	# Displays the dataset
	st.subheader('1. Dataset')


	@st.cache(allow_output_mutation=True)
	def DataToPredict():
		return []

	def ClassesPredict():
		return []

	@st.cache(allow_output_mutation=True)
	def ReadLoadDF(df_file):
		dataframe = pd.read_csv(df_file)
		dataframe.columns = dataframe.columns.str.replace(" ","")
		# df = equalize_events(dataframe)
		# df = neutral_events(populate_ids(populate_NaN(dataframe)))
		return dataframe


	@st.cache
	def convert_df(df):
		return df.to_csv(index=False).encode('utf-8')


	if uploaded_file is not None:
		# dataframe = pd.read_csv(uploaded_file)
		# dataframe.columns = dataframe.columns.str.replace(" ","")
		# df=neutral_events(populate_ids(populate_NaN(dataframe)))
		df = ReadLoadDF(uploaded_file)
		
		st.markdown('**1.1. Glimpse of dataset**')
		chosen_columns = st.multiselect(
			'Exclude/Include column/s',
			df.columns.tolist(),
			default_columns)

		df = df[chosen_columns]
		df = populate_ids(populate_NaN(df))
		df_sampled = df.sample(n=20,random_state=2)
		# st.dataframe(df.sample(n=20,random_state=2))

		df_sampled.to_csv("dfdfdf.csv",index=False,encoding='UTF-8')

		csv = convert_df(df)

		df = equalize_events(df)
		
		st.write(df.EventId.value_counts())


		st.download_button(
			"Press to Download",
			csv,
			"browser_visits.csv",
			"text/csv",
			key='browser-data'
		)


		if classifier == "Random Forest":
			build_forest_model(df[chosen_columns])

		if classifier == "KNN":
			build_knn_model(df[chosen_columns])

	else:
		st.info('Awaiting for CSV file to be uploaded.')





	_, col2, _ = st.columns([1, 4, 7])

	with col2:
		st.write("upload data to evaluate predictions on") # you can use st.header() instead, too

	uploaded_file_for_predict = st.file_uploader("file must be properly structured", type=["csv", "xlsx"])

	# качване на csv съдържащо данни от 14-те канала, тоест без колона EventId
	if uploaded_file_for_predict:
		__for_predict_df = pd.read_csv(uploaded_file_for_predict)
		st.write(__for_predict_df.head())

	dataToPredict = DataToPredict()
	classesToPredict = ClassesPredict()
	import re
	# collect_numbers = lambda x : [float(i) for i in re.split("[^0-9]", x) if i != ""]
	col1,col2 = st.columns([10,2])
	with col1:
		numbers = st.text_area("PLease enter 14 values (separated with ,). THis represent 1 row of BCI data for which we`ll predict if it's left or right.")
		collect_rows = numbers.split('\n')
		collect_numbers = lambda x : [float(i) for i in re.split(",", x) if i != ""]
		st.text('for e.g. #\n4081.0256, 4091.6667, 4069.6155, 4088.4614, 4099.1025, 4092.3076, 4078.7180, 4078.2051, 4066.7949, 4066.4102, 4046.7949, 4050, 4092.9487, 4081.2820')
		
	with col2:
		classes = st.text_area("Enter classes here, each on new row")
		collect_classes = classes.split('\n')
		print(collect_classes)
	col1, _, col2 = st.columns([2,8,2])

	with col1:
		if st.button('append') and len(numbers) > 0:
			for row in collect_rows:
				dataToPredict.append(collect_numbers(row))
			for c in collect_classes:
				classesToPredict.append(c)
	with col2:
		if st.button('clear'):
			dataToPredict.clear()
	try:
		st.table(dataToPredict)
		df_to_predict = pd.DataFrame(dataToPredict)
		st.dataframe(df_to_predict)
	except:
		st.write("!")


	__df = pd.DataFrame(list(dataToPredict),
				columns =["AF3","F7","F3","F5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"])
	print(__df.head())
	# try:
	#     x = zip(*dataToPredict)
	#     st.table(x)
	# except ValueError:
	#     st.title("!")
	predict_from = st.selectbox("Select data source for predictions", ("---", "DF", "Manual Input"))
	if predict_from == "DF":
		# input source is df
		uploaded_file_for_predict
		print('# input source is df')
	if predict_from == "Manual Input":
		# input source is text area
		print('#input source is text area')

	predictions = list()

	if st.button('predict'):
		loaded_knn_model = pickle.load(open("knn_model.pickle", "rb"))
		# predict поредово
		for row in df_to_predict.values:
			# st.text(row)
			# st.text(np.array(row).reshape(1,-1))
			# print(type(np.array(row).reshape(1,-1)))
			prediction = loaded_knn_model.predict(np.array(row).reshape(1,-1))  # np.array(collect_numbers(row)).reshape(1,-1)
			# st.text(prediction)
			
			predictions.append(prediction[0])
		print(type(predictions[0]))

	for x, y in zip(predictions,collect_classes):
		# x, y = str(x, y)
		# if x != y:
		# 	st.write(x, y, "gredhka")
		# else:
		st.write(x, y)





		# for idx, row in enumerate(collect_rows):
		#     print(type(row))
		#     predictions = loaded_knn_model.predict(np.array(collect_numbers(row)).reshape(1,-1))
		#     st.write(f'prediction result {idx}: {predictions}')


		# st.write(predictions.astype(int))
		# implement prediction of the data in the table


	# def main():
	#     pass

if __name__ == "__main__":
    main()