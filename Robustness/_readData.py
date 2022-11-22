import pandas as pd
import numpy as np
import math
import gc
from sklearn.utils import shuffle



def train_valid_test_split(df, test_pct, valid_pct):
  number_of_train_valid_rows = math.floor(df.shape[0] * (1 - test_pct))
  df_train_valid ,df_test = df.iloc[:number_of_train_valid_rows].copy(),df.iloc[number_of_train_valid_rows:].copy() 

  number_of_train_rows = math.floor(df_train_valid.shape[0] * (1 - valid_pct))
  df_train, df_valid = df_train_valid.iloc[:number_of_train_rows].copy(),df_train_valid.iloc[number_of_train_rows:].copy() 

  return df_train, df_valid, df_test


def get_val_idx(df, valid_pct):
    row_cutoff = math.ceil(df.shape[0] * (valid_pct))
    return list(range(0, row_cutoff))

def getAndPrepareData():
	fixed_seed = 42
	dataPath = '../ReducedFiles/'

	diboson     = pd.read_pickle(dataPath +'diboson.pkl')
	DYee     = pd.read_pickle(dataPath +'DYee.pkl')
	DYmumu     = pd.read_pickle( dataPath  + 'DYmumu.pkl')
	DYtautau    = pd.read_pickle( dataPath  + 'DYtautau.pkl')
	ttbar_lep     = pd.read_pickle( dataPath  + 'ttbar_lep.pkl')
	Wenu     = pd.read_pickle( dataPath  + 'Wenu.pkl')
	Wmunu     = pd.read_pickle( dataPath  + 'Wmunu.pkl')
	Wtaunu     = pd.read_pickle( dataPath  + 'Wtaunu.pkl')
	Zee     = pd.read_pickle( dataPath  + 'Zee.pkl')
	ttbar_had     = pd.read_pickle( dataPath  + 'ttbar_had.pkl')
	Ztautau     = pd.read_pickle( dataPath  + 'Ztautau.pkl')
	signal     = pd.read_pickle(dataPath  + 'signal.pkl')

	diboson     = diboson.loc[:,~diboson.columns.duplicated()]
	DYee        = DYee.loc[:,~DYee.columns.duplicated()]
	DYmumu      = DYmumu.loc[:,~DYmumu.columns.duplicated()]
	DYtautau    = DYtautau.loc[:,~DYtautau.columns.duplicated()]
	ttbar_lep   = ttbar_lep.loc[:,~ttbar_lep.columns.duplicated()]
	Wenu        = Wenu.loc[:,~Wenu.columns.duplicated()]
	Wmunu       = Wmunu.loc[:,~Wmunu.columns.duplicated()]
	Wtaunu      = Wtaunu.loc[:,~Wtaunu.columns.duplicated()]
	Zee         = Zee.loc[:,~Zee.columns.duplicated()]
	ttbar_had   = ttbar_had.loc[:,~ttbar_had.columns.duplicated()]
	Ztautau     = Ztautau.loc[:,~Ztautau.columns.duplicated()]
	signal      = signal.loc[:,~signal.columns.duplicated()]

	diboson = shuffle(diboson, random_state = 42)
	DYee = shuffle(DYee, random_state = 42)
	DYmumu = shuffle(DYmumu, random_state = 42)
	DYtautau = shuffle(DYtautau, random_state = 42)
	ttbar_lep = shuffle(ttbar_lep, random_state = 42)
	Wenu = shuffle(Wenu, random_state = 42)
	Wmunu = shuffle(Wmunu, random_state = 42)
	Wtaunu = shuffle(Wtaunu, random_state = 42)
	Zee = shuffle(Zee, random_state = 42)
	ttbar_had = shuffle(ttbar_had, random_state = 42)
	Ztautau = shuffle(Ztautau, random_state = 42)
	signal = shuffle(signal, random_state = 42)

	background = [diboson, DYee, DYmumu, DYtautau, ttbar_lep, Wenu, Wmunu, Wtaunu, Zee, ttbar_had, Ztautau]
	backgroundLabel = ['diboson', 'DYee', 'DYmumu', 'DYtautau', 'ttbar_lep', 'Wenu', 'Wmunu', 'Wtaunu', 'Zee', 'ttbar_had', 'Ztautau']

	del Zee
	del ttbar_lep
	del DYmumu
	del DYee
	del diboson
	del Wenu
	del Ztautau
	gc.collect()


	features_from_feature_importance = [
										"met_et",
										"lep_1_E",
										"lep_2_E",
										"lep_3_E",
										"lep_1_eta",
										"lep_2_eta",
										"jet_n",
										"lep_1_pt",
										"lep_2_pt",
										"lep_3_pt",
										"lep_4_pt",
										"lep_5_pt",
										"lep_1_phi",
										"lep_2_phi",
										"jet_2_trueflav",
										"jet_1_E",
										"jet_3_E",
										"jet_1_pt",
										"jet_2_pt",
										"jet_3_pt",
										"jet_4_pt",
										"jet_5_pt",
										"jet_6_pt",
										"jet_7_pt",
										"jet_8_pt",
										"jet_9_pt",
										"alljet_n",
										"lep_1_etcone20",
										"jet_2_MV1",
										"jet_1_MV1",
										"jet_1_phi",
										"jet_1_m",
										"jet_2_E",
										"jet_2_jvf",
										"jet_1_SV0",
										]

	invariant_features = [ 'lep_1_pt',
						'lep_1_eta',
						'lep_1_phi',
						'lep_1_type',
						'lep_1_charge',
						'lep_1_E',
						'lep_2_pt',
						'lep_2_eta',
						'lep_2_phi',
						'lep_2_type',
						'lep_2_charge',
						'lep_2_E',
						'jet_1_pt',
						'jet_1_eta',
						'jet_1_phi',
						'jet_2_pt',
						'jet_2_eta',
						'jet_2_phi',
			
	]

	features_and_weights = list(set(features_from_feature_importance + invariant_features))
	features = features_and_weights

	for i in range(0,11):
		background[i]['data_type'] = 0

	signal['data_type'] = 1

	features.append('data_type')

	background_train = [None]*len(background)
	background_valid = [None]*len(background)
	background_test = [None]*len(background)
	background_train_valid = [None]*len(background)


	for i in range(len(background)):
		background[i].reset_index()
		background_train[i], background_valid[i], background_test[i] = train_valid_test_split(background[i], 0.6, 0.3)
		background_train_valid[i] = pd.concat([background_train[i][features], background_valid[i][features]])


	signal.reset_index()
	signal_train, signal_valid, signal_test = train_valid_test_split(signal, 0.6, 0.3)
	signal_train_valid = pd.concat([signal_train[features], signal_valid[features]])

	df_train_valid = pd.concat(
		[background_train_valid[0][features],
		background_train_valid[1][features],
		background_train_valid[2][features],
		background_train_valid[3][features],
		background_train_valid[4][features],
		background_train_valid[5][features],
		background_train_valid[6][features],
		background_train_valid[7][features],
		background_train_valid[8][features],
		background_train_valid[9][features],
		background_train_valid[10][features],
		signal_train_valid[features]]
	)

	df_test =  pd.concat(
		[background_test[0][features],
		background_test[1][features],
		background_test[2][features],
		background_test[3][features],
		background_test[4][features],
		background_test[5][features],
		background_test[6][features],
		background_test[7][features],
		background_test[8][features],
		background_test[9][features],
		background_test[10][features],
		signal_test[features]]
	)
	df_train_valid.reindex(columns=features)
	features.remove('data_type')


	#Normalizing the train_valid dataset
	[df_train_valid[col].update((df_train_valid[col] - df_train_valid[col].min()) / (df_train_valid[col].max() - df_train_valid[col].min())) for col in df_train_valid[features].columns]
	#Normalizing the test dataset
	[df_test[col].update((df_test[col] - df_test[col].min()) / (df_test[col].max() - df_test[col].min())) for col in df_test[features].columns]


	df_train_valid = df_train_valid.sample(frac=1, random_state=fixed_seed).reset_index(drop=True)
	df_train_valid["data_type"] = df_train_valid["data_type"].astype("category")

	indexes = get_val_idx(df_train_valid, 0.4)
	df_train = df_train_valid.iloc[indexes[-1]:] 
	df_valid = df_train_valid[:indexes[-1]]

	return df_train, df_valid, df_test

def getData():
    return getAndPrepareData()

def saveToFile(df_train, df_valid, df_test):
	df_train.to_csv('../ReducedFiles/df_train.csv')
	df_valid.to_csv('../ReducedFiles/df_valid.csv')
	df_test.to_csv('../ReducedFiles/df_test.csv')

#saveToFile( df_train, df_valid, df_test)

def getDataFromFile():
    df_train = pd.read_csv('../ReducedFiles/df_train.csv', index_col=0)
    df_valid = pd.read_csv('../ReducedFiles/df_valid.csv', index_col=0)
    df_test = pd.read_csv('../ReducedFiles/df_test.csv', index_col=0)
    return df_train, df_valid, df_test
