from tqdm import tqdm
import pandas as pd

if "globglob" not in dir():
    from glob import glob as globglob
#
if "ossep" not in dir():
    from os import sep as ossep
#
if "sub" not in dir():
    from re import sub
#
if "csv_reader" not in dir():
    from csv import reader as csv_reader
#

### import all 
def set_up_environment(aethpy_path=None):
	import os, re
	from functools import reduce

	import pandas as pd
	import numpy as np

	import plotly.express as px
	import plotly.io as pio
	pio.renderers.default = "vscode"
	import matplotlib.pyplot as plt

	# Keep path configuration explicit and environment-driven.
	if aethpy_path is None:
		aethpy_path = os.environ.get("AETHPY_SRC", "")
	if aethpy_path:
		os.environ["AETHPY_SRC"] = aethpy_path
	return

# import all the data from a directory assuming QAQC type formatting
def import_MA_test_data(data_dir):
	## now load individual modules from aethpy, e.g.
	from aethpy import fileHandle as apyfh
	import pandas as pd

	calibration_dir = data_dir

	df =\
		apyfh.readall_BCdata_from_dir(
			directory_path=\
				calibration_dir,
			sep=',',
			mult_folders_in_dir = True,
			verbose = False,
			summary = False,
			AE51_devices_only = False,
			file_number_printout = True,
			output_pax_averaged_by_minute = True,
			PAX_correction = False,
			inter_device_corr = True,
			assign_testid_from_startdate = True,
			assign_testid_from_filename = False,
			num_mins_datetime_round = 1,
			group_Session_Ids_together = False, 
			datetime_fixme_dict = {},
			assign_unit_category_from_dirname = False,
			test_batch_indication = False,
			allow_ALx_files = True,
			files_to_exclude = [],
			output_first_datapoint_of_each_file_separately = False,
			create_session_ids = True,
			process_api_formatted_files=False,
			sigmaatn_al30 = 12.5/1.55,
			sigmaatn_al60 = 12.5/1.8,
			sigmaatn_al80 = 12.5/1,
		)

	print('\n\nDropping duplicates')
	## Drop MAx duplicates based on Serial number and Datum ID
	df_ma = df.loc[df['Serial number'].str.contains("MA")].drop_duplicates(subset=['Serial number', 'Datum ID'], keep='first')
	## Drop duplicates in other instruments (which do not have a Datum ID variable) based on Serial number and datetime_local
	df_others = df.loc[(df['Serial number'].str.contains("MA") == False)].drop_duplicates(subset=['Serial number', 'datetime_local'], keep='first')
	## Concatenate the two duplicates-dropped dataframes
	df = pd.concat([df_ma, df_others], ignore_index=True)

	## sort
	df = df.sort_values(by=['Serial number', 'datetime_local']).reset_index(drop=True)
	return df

# remove tape advances
# def remove_tape_advance(df):
# 	df_nonviable_statuses =\
# 	df.loc[
# 		(df['Readable status'].str.contains('Start up') == True) |
# 		(df['Readable status'].str.contains('Tape advance') == True),
# 	: ].reset_index(drop=True)

# 	# Remove Start up and Tape advance data from main dataframe, as BC data are not useful
# 	df =\
# 		df.loc[
# 			(df['Readable status'].str.contains('Start up') == False) &
# 			(df['Readable status'].str.contains('Tape advance') == False),
# 		: ].reset_index(drop=True)
# 	#
# 	print("%s datapoints removed due to Start up or Tape advance status" % df_nonviable_statuses.shape[0])
# 	return df

def remove_tape_advance(df):
    pattern = 'Start up|Tape advance'

    # Identify rows with unwanted statuses
    df_nonviable_statuses = df.loc[
        df['Readable status'].str.contains(pattern, na=False),
        :
    ].reset_index(drop=True)

    # Keep only rows that do NOT contain those statuses
    df_clean = df.loc[
        ~df['Readable status'].str.contains(pattern, na=False),
        :
    ].reset_index(drop=True)

    print(f"{df_nonviable_statuses.shape[0]} datapoints removed due to Start up or Tape advance status")

    return df_clean

# remove concerning statuses
def remove_concerning_statuses(df):
	df['Readable status'].fillna('Time source manual', inplace=True)
	### Concerning statuses
	## Define concerning statuses
	concerning_readable_status_values =\
		[
			'Flow unstable', 'Optical saturation',
			'Sample timing error', 
			# 'User skipped tape advance'
		]
	concerning_statuses_as_regex_string =\
		'|'.join(concerning_readable_status_values)
	## Isolate datapoints with concerning statuses & retain for later diagnostics
	print("Statuses of concern, count by device and status:\n")
	df_concerning_statuses =\
		df.loc[
			df['Readable status'].str.contains(concerning_statuses_as_regex_string, na=False),
		: ].reset_index(drop=True)
	for sn in df_concerning_statuses['Serial number'].unique():
		for status in concerning_readable_status_values:
			print(sn, status, df_concerning_statuses.loc[df_concerning_statuses['Serial number'] == sn, 'Readable status'].astype(str).str.contains(status).sum())
	## Drop data with concerning statuses from primary dataframe
	df =\
		df.loc[
			df['Readable status'].str.contains(concerning_statuses_as_regex_string) == False,
		: ].reset_index(drop=True)
	return df

# clean optics
def check_optics(df):
	original_length = len(df.index)
	### Check for invalid ref and sen values not captured by Optical saturation status value
	max_optics = 2**20
	# ID such datapoints
	df_invalid_optics =\
		df.loc[
			(df['IR Ref'] >= max_optics) |
			(df['IR Sen1'] >= max_optics) |
			(df['IR Sen2'] >= max_optics) |
			(df['Red Ref'] >= max_optics) |
			(df['Red Sen1'] >= max_optics) |
			(df['Red Sen2'] >= max_optics) |
			(df['Green Ref'] >= max_optics) |
			(df['Green Sen1'] >= max_optics) |
			(df['Green Sen2'] >= max_optics) |
			(df['Blue Ref'] >= max_optics) |
			(df['Blue Sen1'] >= max_optics) |
			(df['Blue Sen2'] >= max_optics) |
			(df['UV Ref'] >= max_optics) |
			(df['UV Sen1'] >= max_optics) |
			(df['UV Sen2'] >= max_optics)
		]
	# Remove such datapoints
	df =\
		df.loc[
			(df['IR Ref'] < max_optics) &
			(df['IR Sen1'] < max_optics) &
			(df['IR Sen2'] < max_optics) &
			(df['Red Ref'] < max_optics) &
			(df['Red Sen1'] < max_optics) &
			(df['Red Sen2'] < max_optics) &
			(df['Green Ref'] < max_optics) &
			(df['Green Sen1'] < max_optics) &
			(df['Green Sen2'] < max_optics) &
			(df['Blue Ref'] < max_optics) &
			(df['Blue Sen1'] < max_optics) &
			(df['Blue Sen2'] < max_optics) &
			(df['UV Ref'] < max_optics) &
			(df['UV Sen1'] < max_optics) &
			(df['UV Sen2'] < max_optics)
		].reset_index(drop=True)
	#
	print(
		"Number of datapoints with invalid optics values\nAFTER dropping data with 'Optical saturation' status values: %s" % df_invalid_optics.shape[0]
	)
	print(f"Removed {original_length - len(df.index)} datapoints for optics")
	return df

# run data cleaning functions
def clean_data(df):
	
	df = remove_tape_advance(df)
	df = remove_concerning_statuses(df)
	df = check_optics(df)
	return df

# add delta temperature and atn and other env variables to df
# def add_deltas(calibration_df, rolling_window_number_of_timebases = 10):
# 	for col in calibration_df:
# 		if 'delta ' in col or ' rolling mean' in col:
# 			del calibration_df[col]

# 	### Create delta variables of interest
# 	atns = []
# 	for col in calibration_df.columns:
# 		if 'ATN' in col:
# 			atns.append(col)
			
# 	for delta_var_i in (atns +['Internal temp (C)', 'Sample temp (C)', 'Sample RH (%)']):

# 		new_delta_var_name_i = 'delta ' + delta_var_i
# 		new_delta_var_name_rolling_i = 'delta ' + delta_var_i + ' rolling mean'
# 		new_var_name_rolling_i = delta_var_i + ' rolling mean'
# 		# Create delta variable
# 		# df_i =\
# 		# 	calibration_df.set_index("datetime_local").groupby(['Serial number','Session ID','Tape position']).apply(lambda x: x[delta_var_i].diff()).reset_index(name=new_delta_var_name_i)
# 		# calibration_df = calibration_df.merge(
# 		# 	df_i,
# 		# 	on=['Serial number','Session ID','Tape position','datetime_local']
# 		# )
# 		df_i = (
# 			calibration_df
# 			.set_index("datetime_local")
# 			.groupby(['Serial number', 'Session ID', 'Tape position'],  group_keys=False)
# 			.apply(lambda x: x[delta_var_i].diff())
# 			.reset_index()
# 			#.rename(columns={delta_var_i: new_delta_var_name_i})
# 		)

# 		print(df_i.head(5))

# 		calibration_df = calibration_df.merge(
# 			df_i,
# 			on=['Serial number', 'Session ID', 'Tape position', 'datetime_local']
# 		)

# 		# # Create delta variable rolling mean smooth
# 		# df_j =\
# 		# 	calibration_df.set_index("datetime_local").groupby(['Serial number','Session ID','Tape position']).apply(lambda x: x[delta_var_i].diff()).rolling(rolling_window_number_of_timebases).mean().reset_index(name=new_delta_var_name_rolling_i)
# 		# calibration_df = calibration_df.merge(
# 		# 	df_j,
# 		# 	on=['Serial number','Session ID','Tape position','datetime_local']
# 		# )

# 		df_j = (
# 			calibration_df
# 			.set_index("datetime_local")
# 			.groupby(['Serial number', 'Session ID', 'Tape position'])
# 			.apply(lambda x: x[delta_var_i].diff())
# 			.rolling(rolling_window_number_of_timebases)
# 			.mean()
# 			.reset_index()
# 			.rename(columns={delta_var_i: new_delta_var_name_rolling_i})
# 		)

# 		calibration_df = calibration_df.merge(
# 			df_j,
# 			on=['Serial number', 'Session ID', 'Tape position', 'datetime_local']
# 		)

# 		# Create variable rolling mean smooth
# 		df_k = (
# 			calibration_df.set_index("datetime_local")
# 			.groupby(['Serial number', 'Session ID', 'Tape position'])[delta_var_i]  # Specify the column to apply rolling mean
# 			.rolling(rolling_window_number_of_timebases)
# 			.mean()
# 			.reset_index()  # This will include the index in the reset DataFrame
# 			.rename(columns={delta_var_i: new_var_name_rolling_i})
# 		)

# 		calibration_df = calibration_df.merge(
# 			df_k,
# 			on=['Serial number', 'Session ID', 'Tape position', 'datetime_local'],
# 		)

# 	return calibration_df


# new function circa May. 2025 Kyan Shlipak
def add_deltas(calibration_df, rolling_window_number_of_timebases=10):
    # Step 1: Remove existing delta and rolling mean columns
    cols_to_remove = [col for col in calibration_df.columns if 'delta ' in col or ' rolling mean' in col]
    calibration_df.drop(columns=cols_to_remove, inplace=True, errors='ignore')

    # Step 2: Identify ATN columns and other variables of interest
    atns = [col for col in calibration_df.columns if 'ATN' in col]
    other_vars = ['Internal temp (C)', 'Sample temp (C)', 'Sample RH (%)']
    delta_vars = atns + other_vars

    # Step 3: Create delta variables
    for delta_var in delta_vars:
        # Create delta variable
        calibration_df[f'delta {delta_var}'] = (
            calibration_df.groupby(['Serial number', 'Session ID', 'Tape position'])[delta_var]
            .diff()
        )

        # Create delta variable rolling mean
        delta_mean = (
            calibration_df.groupby(['Serial number', 'Session ID', 'Tape position'])[f'delta {delta_var}']
            .rolling(rolling_window_number_of_timebases, min_periods=1)
            .mean()
        )
        # Assign the rolling mean to a new column
        calibration_df[f'delta {delta_var} rolling mean'] = delta_mean.reset_index(level=[0, 1, 2], drop=True)

        # Create variable rolling mean
        var_mean = (
            calibration_df.groupby(['Serial number', 'Session ID', 'Tape position'])[delta_var]
            .rolling(rolling_window_number_of_timebases, min_periods=1)
            .mean()
        )
        # Assign the rolling mean to a new column
        calibration_df[f'{delta_var} rolling mean'] = var_mean.reset_index(level=[0, 1, 2], drop=True)

    return calibration_df

# for calibration, needs to contain a flow factor variable (Flow1 factor and Flow2 factor)
def bc_calc(dataframe,
			output_var_bc,
			input_deltaatn_var,
			input_flow_var,
			input_deltatime_var,
			sigmaatn_ma,
			filter_surfacearea = 0.071):
	df = dataframe.copy(deep=True)

	# scaler to unify temporal units in flowrate and timebase from seconds to minutes
	flow_timebase_temporal_unity_scaler = 1/60

	sigma_scaler = 1000000/1000000000


	# scaler to convert flow units from mL to m3
	flow_volume_scaler = 1/1000000 # convert to m3 per minute

	# for MA units
	a1 =\
		df.loc[df["Serial number"].str.contains('MA'), input_deltaatn_var] / (sigmaatn_ma * sigma_scaler) # convert sigma units from m2/g to mm2/ng
		# units = (ATN * ng) / mm2 = ng/mm2

	flow = df.loc[df["Serial number"].str.contains('MA'), input_flow_var] * flow_volume_scaler # convert to L/s for consistency with timebase being in seconds (and flow being in per-minute); convert mL to m3
	deltatime = df.loc[df["Serial number"].str.contains('MA'), input_deltatime_var] * flow_timebase_temporal_unity_scaler
	b1 = filter_surfacearea / (flow * deltatime) # units = mm2 / m3
	bc = a1*b1 # units = ng/mm2 * mm2/m3 = ng/m3

	spot = int(output_var_bc[-1])
	
	flow_factor = df[f'Flow{spot} Factor']
	# if spot == 1:
	# 	flow_factor = reference_flow1  / df[f'Flow1 (mL/min)']
	# if spot == 2:
	# 	flow_factor = reference_flow2  / df[f'Flow2 (mL/min)']

	bc = bc / flow_factor
	df.loc[df["Serial number"].str.contains('MA'), output_var_bc] = bc

	return df[output_var_bc]

# do standard BC calculations
def perform_first_principle_calcs_corrected(
		dataframe,
		wl='IR',
		dev_type='MA200',
		flow_var1='Flow1 (mL/min)',
		flow_var2='Flow2 (mL/min)',
		al30_sigma = 6.776,
		al60_sigma = 12.5*1,
		al80_sigma = 12.5*1,
		delta_time_var = 'delta_time'
	):
	
	'''
	Take a dataframe and recreate various aspects of firmware code to reproduce intermediary and other first principle outputs.

	Returns original dataframe with appended columns.

	Spot size is radius in mm
	'''

		## now load individual modules from aethpy, e.g.
	from aethpy import dataParse as apydp
	from aethpy import firstPrinciples as apyfp

	df = dataframe.copy(deep=True)

	# ATN cross section dict is for MAx only!
	atn_cross_section_dict_MAx = {
			'Blue':19.070,
			'Green':17.028,
			'Red':14.091,
			'IR': 10.120,
			'UV':24.069
		}

	# Set 
	if dev_type in ['MA200','MA300','MA350']:
		atn_cross_section_dict = atn_cross_section_dict_MAx
		sigmaatn = atn_cross_section_dict[wl]
		spotsize = 3
		create_unique_df_fxn = apydp.create_df_list_atlevel_sid_tapePosition
		if "%s BC2" %wl in df.columns:
			spot_list = [1,2]
		else:
			spot_list = [1]
	elif dev_type in ['AL30']:
		sigmaatn = al30_sigma
		create_unique_df_fxn = apydp.create_df_list_atlevel_sid
		spotsize = 3
		spot_list = [1]


	for spot in spot_list:
		if spot ==1:
			flow_var = flow_var1
		if spot == 2:
			flow_var = flow_var2
		spot = str(spot)

		atn_var = 'Corrected ' +  wl + ' ATN' + spot

		# print('Calculating optical variables from first principles for:')
		df_interim_list = []
		for df_interim in create_unique_df_fxn(df):
			# Adjust ATN to 0
			df_interim.sort_values(by='datetime_local', inplace=True, ignore_index=True)
			start_index = 0
			while True:
				starting_atn = df_interim[atn_var].iloc[start_index]
				if pd.isna(starting_atn):
					start_index += 1
				else:
					break
			df_interim[atn_var] = df_interim[atn_var] - starting_atn

			# dATN
			datn_var =\
				'Corrected delta ' + wl + ' ATN' + spot

			# bc calc
			bc_var = 'Corrected ' + wl + ' BC' + spot


			df_interim[bc_var] =\
				bc_calc(
				dataframe = df_interim,
				output_var_bc = bc_var,
				input_deltaatn_var = datn_var,
				input_flow_var = flow_var,
				input_deltatime_var = 'delta_time',
				sigmaatn_ma = sigmaatn,
				filter_surfacearea = 0.0713)

			# append to list of df_interims
			df_interim_list.append(df_interim)

		# reconcatenate all processed data
		df = pd.concat(df_interim_list)

	# k and BCc calc
	if 2 in spot_list:
		bc1_var = 'Corrected '+ wl + ' BC1' 
		bc2_var = 'Corrected '+ wl + ' BC2' 
		atn1_var = 'Corrected '+ wl + ' ATN1'
		atn2_var = 'Corrected '+ wl + ' ATN2'
		k_var = 'Corrected k ' + wl 
		bcc_var = 'Corrected '+ wl + ' BCc'
		#
		df[k_var] = apyfp.k_calc(dataframe= df, bc1=bc1_var, bc2=bc2_var, atn1=atn1_var,atn2=atn2_var)
		#
		df[bcc_var] = apyfp.bcc_calc(dataframe = df, bc1=bc1_var,k=k_var,atn1=atn1_var)

	return df



def convert_to_float(df_cleaned_status):
	import numpy as np
	"""
	This function converts all columns in df_cleaned_status that are of type 'object' to 'float'.
	It returns the modified DataFrame.
	Args:
		df_cleaned_status (pd.DataFrame): The DataFrame to be modified.
	Returns:
		pd.DataFrame: The modified DataFrame with object columns converted to float.
	"""
	wls = ['IR', 'UV', 'Blue', 'Green', 'Red']
	vars = ['ATN1', 'ATN2', 'BCc', 'BC1', 'BC2', 'Sen1', 'Sen2', 'Ref', 'K']
	for wl in wls:
		for var in vars:
			col_name = f"{wl} {var}"
			if col_name in df_cleaned_status.columns and df_cleaned_status[col_name].dtype == 'object':
				try:
					df_cleaned_status.loc[df_cleaned_status[col_name] == '"', col_name] = np.nan  # Replace 'NaN' strings with None
					df_cleaned_status[col_name] = df_cleaned_status[col_name].astype(float)
					print(f"Converted {col_name} to float.")
				except ValueError:
					print(f"Could not convert {col_name} to float due to non-numeric values.")	
	return df_cleaned_status

def resample_df(df_for_regression_MA, timebase='5min', mean = True):
    import pandas as pd

    # Set the datetime column as index for resampling
    df_for_regression_MA = df_for_regression_MA.set_index('datetime_local')

    # Split the DataFrame into numeric and non-numeric columns
    numeric_cols = df_for_regression_MA.select_dtypes(include='number')
    non_numeric_cols = df_for_regression_MA.select_dtypes(exclude='number')

    # Group by 'Serial number' and resample the numeric columns
    # Here we don't forward-fill or interpolate; we just resample and keep NaNs where no data exists
    if mean:
        resampled_numeric = numeric_cols.groupby(df_for_regression_MA['Serial number']).resample(timebase).mean().reset_index(level=0, drop=True)
    else: 
        resampled_numeric = numeric_cols.groupby(df_for_regression_MA['Serial number']).resample(timebase).median().reset_index(level=0, drop=True)

    try:
        resampled_numeric = resampled_numeric[~resampled_numeric['IR BCc'].isna()]
    except Exception as e:
        print("Error while checking for NaN in 'IR BCc':", e)
    # We no longer forward fill the non-numeric columns, just resample them and leave missing values as NaN
    non_numeric_resampled = non_numeric_cols.groupby(df_for_regression_MA['Serial number']).resample(timebase).first().reset_index(level=0, drop=True)
    non_numeric_resampled = non_numeric_resampled[~non_numeric_resampled['Serial number'].isna()]   
    # Combine the resampled numeric and non-numeric columns (no filling)
    df_resampled = pd.concat([resampled_numeric, non_numeric_resampled], axis=1)

    # Reset index for the final DataFrame, if needed
    df_resampled.reset_index(inplace=True)

    return df_resampled



def histogram_plot(df, col, bins_lower = False, bins_upper = False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Drop NaNs
    data = df[col].dropna()

    # Define bins within [-0.2, 0.2]

    plt.figure(figsize=(8, 5))

    if bins_lower and bins_upper:
        bins = np.linspace(bins_lower, bins_upper, 50)  # 39 bins between -0.2 and 0.2
        sns.histplot(data, bins=bins, kde=False, color='skyblue', edgecolor='black')
    else:
        sns.histplot(data, bins=41, kde=False, color='skyblue', edgecolor='black')

    plt.title(f'Histogram of {col}')
    plt.xlabel(f'{col}')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()



# filter data to that which is suitable for temperature compensation analyses
def filter_for_temperature_compensation(df, threshold = 750, window = 25, cut_off = 5, exclude_last_hour = True):
	import pandas as pd
	import numpy as np
	# Step 1: Average Data Across Serial Numbers
	df_avg = df.groupby(['datetime_local', 'test'], as_index=False)['IR BC1 smoothed  (ng/m^3)'].mean()
	df_avg['below_threshold'] = df_avg['IR BC1 smoothed  (ng/m^3)'] < threshold
	df_avg.sort_values(by=['test', 'datetime_local'], inplace=True)

	# Step 3: Exclude Data from the Last Hour of Each Test
	def exclude_last_hour_func(group):
		last_hour_start = group['datetime_local'].max() - pd.Timedelta(hours=1)
		return group[group['datetime_local'] < last_hour_start]

	# Apply the exclusion function
	if exclude_last_hour:
		df_avg = df_avg.groupby('test').apply(exclude_last_hour_func).reset_index(drop=True)

	# Identify continuous periods below threshold
	def identify_periods(group):
		# Identify changes in below_threshold column
		group['change'] = group['below_threshold'].astype(int).diff().fillna(0).astype(int)
		
		# Identify period start and end
		group['period_id'] = (group['change'] != 0).cumsum()
		
		# Calculate period durations
		period_durations = group.groupby('period_id').agg(
			start_time=('datetime_local', 'min'),
			end_time=('datetime_local', 'max'),
			duration=('datetime_local', lambda x: (x.max() - x.min()).total_seconds() / 60),
			below_threshold=('below_threshold', 'any')
		)
		
		# Filter periods with duration >= 20 minutes and below_threshold == True
		valid_periods = period_durations[(period_durations['duration'] >= window) & (period_durations['below_threshold'])]
		# Filter original DataFrame to include only valid periods
		valid_times = valid_periods[['start_time', 'end_time']].reset_index(drop=True)
		mask = np.zeros(len(group), dtype=bool)
		for _, row in valid_times.iterrows():
			mask |= (group['datetime_local'] >= row['start_time']) & (group['datetime_local'] <= row['end_time'])
		
		return group[mask]

	# Apply the period identification function
	df_filtered = df_avg.groupby('test').apply(identify_periods).reset_index(drop=True)
	df_filtered = df[df['datetime_local'].isin(df_filtered['datetime_local'])]

	df_filtered = df_filtered.loc[abs(df_filtered['delta IR ATN1']) <= 0.5]
	
	# Calculate the cutoff time (max datetime_local minus 5 minutes)
	cutoff_time = df_filtered['datetime_local'].max() - pd.Timedelta(minutes=cut_off)

	# Filter the DataFrame to remove the last 5 minutes of data
	df_filtered = df_filtered[df_filtered['datetime_local'] < cutoff_time]
	return df_filtered

# regress dATN/dT with HEPA testing to get temperature compensation factors
def generate_temperature_regression(df, atten_var, x_col = 'delta Sample temp (C) rolling mean', y_col = 'delta IR ATN1 rolling mean', datn_threshold = 0.5):
	import plotly.express as px
	import statsmodels.api as sm
	import pandas as pd

	# Specify the list of devices to exclude
	exclude_names = []#['MA350-0525',  'MA200-0267']

	# Select all rows where 'Serial number' is not in the exclude_names list
	df_filtered = df[~df['Serial number'].isin(exclude_names)]
	df_filtered = df_filtered.dropna(subset=[x_col, y_col])
	df_filtered = df_filtered.loc[abs(df_filtered[y_col]) <= datn_threshold]

		# Initialize lists to store results
	serial_numbers = []
	slopes = []
	r_squared_values = []

	# Group by 'Serial number'
	for serial, group in df_filtered.groupby(['sn_session_id']):
		try:
			x = group[x_col]
			y = group[y_col]
			
			# Perform the OLS regression
			model = sm.OLS(y, sm.add_constant(x))
			results = model.fit()
			
			# Extract the slope and R-squared values
			slope = results.params[1]
			intercept = results.params[0]  # Intercept (if x includes a constant term)
			r_squared = results.rsquared
			
			# Append results
			serial_numbers.append(serial)
			slopes.append(slope)
			r_squared_values.append(r_squared)
		except Exception as e:
			print(f"Failure for serial number {serial}")

	# Create a DataFrame for regression results
	regression_results_df = pd.DataFrame({
		'Serial number': serial_numbers,
		'Slope': slopes,
		'R-squared': r_squared_values
	})

	# Define the variables
	x = df_filtered[x_col]
	y = df_filtered[y_col]

	# Perform the OLS regression
	model = sm.OLS(y, sm.add_constant(x))
	results = model.fit()

	# Extract the slope and R-squared values
	slope = results.params[1]
	intercept = results.params[0]  # Intercept (if x includes a constant term)
	r_squared = results.rsquared

	wavelength = atten_var.split(' ')[0]
	number = atten_var[-1]

	# Create the scatter plot with OLS trendline
	fig1 = px.scatter(
		df_filtered,
		x=x_col,
		y=y_col,
		trendline='ols',
		title=f"{x_col} vs. {y_col}<br>Slope: {slope:.7f}, R²: {r_squared:.4f}",
		height=600,
		width=700,
		color = 'sn_session_id',
		hover_data=['Tape position', 'datetime_local', f'{wavelength} BC{number}', atten_var],
		opacity=0.6,
	).update_xaxes(matches=None, showticklabels=True).update_yaxes(matches=None, showticklabels=True)


	fig2 = px.scatter(
		df_filtered,
		x=x_col,
		y=y_col,
		trendline='ols',
		title=f"{x_col} vs. {y_col}<br>Slope: {slope:.7f}, R²: {r_squared:.4f}",
		height=600,
		width=700,
		#color = 'Serial number',
		hover_data=['Tape position', 'datetime_local', f'{wavelength} BC{number}', atten_var],
		opacity=0.6,
	).update_xaxes(matches=None, showticklabels=True).update_yaxes(matches=None, showticklabels=True)
	# Show the plot
	#fig.show()

	return fig1, slope, r_squared, regression_results_df, fig2

# correct with temperature compensation factors
def correct_temperature_compensation(df, constant_1 = 0.052, constant_2 = 0.018, max = 0.8):
	print("cor")
	# Function to apply correction for each device
	def correct_ir_atn(group, var, correction):
		group = group.reset_index(drop = True)
		#group[f'delta {var}'] = group[var].diff()
		group['delta Sample Temp (C)'] = group['Sample temp (C)'].diff()
		
		#print(group[f'Corrected delta {var}'].iloc[0:5])
		group[f'Corrected delta {var}'] = group[f'delta {var}']
		group[f'Corrected delta {var}'].iloc[1:] = group[f'delta {var}'].iloc[1:] - correction * group['delta Sample Temp (C)'].iloc[1:]
		group.loc[abs(group[f'Corrected delta {var}']) > max, f'Corrected delta {var}'] = 0

		group[f'Corrected {var}'] = group[var]
		group[f'Corrected {var}'].iloc[1:] = group[var].iloc[0] + group[f'Corrected delta {var}'].iloc[1:].cumsum()

		return group


	df = df.groupby(['sn_session_id']).apply(lambda group: correct_ir_atn(group, "IR ATN1", constant_1)).reset_index(drop = True)
	df = df.groupby(['sn_session_id']).apply(lambda group: correct_ir_atn(group, "IR ATN2", constant_2)).reset_index(drop = True)
	
	df = df.sort_values(by='datetime_local')

	for device in df['Serial number'].unique():
		df.loc[df['Serial number'] == device, 'delta_time'] = \
			df.loc[df['Serial number'] == device]['datetime_local'].diff().dt.total_seconds()

	df['Flow1 Factor'] = 1
	df['Flow2 Factor'] = 1

	device_type = df['device_type'].unique()[0]
	
	df_corrected = perform_first_principle_calcs_corrected(#perform_first_principle_calcs_sanity(
		df,
		wl="IR",
		dev_type="MA350",
		flow_var1='Flow1 (mL/min)',
		flow_var2='Flow2 (mL/min)',
		al30_sigma = 6.776,
		al60_sigma = 12.5*1,
		al80_sigma = 12.5*1,
		delta_time_var = 'delta_time'
	)
	
	return df_corrected


## Problem solved here: trying to optimize a temperature compensation factor given unfiltered data from outdoor monitors
# solution is to minimize error between average of red and green signals and the IR signal
def optimize_correction_factor(df, target_spot='BC1', min = 0.02, max = 0.1, num = 15, other_constant = 0.018, resample_timebase = '15min'):
	import numpy as np
	import pandas as pd
	target_column = f"RG {target_spot}"
	if target_spot == "BC1":
		comparison_columns = ['Corrected IR BC1', 'IR BC1']

	if target_spot == "BC2":
		comparison_columns = ['Corrected IR BC2', 'IR BC2']

	# Range of constant_1 values to iterate over
	constant_values = np.linspace(min, max, num)  # Example: 15 values from 0.02 to 0.1

	# DataFrame to store results
	results = []

	# Iterate over each value of constant_1
	for constant in constant_values:
		# Apply the temperature compensation function
		if target_spot == "BC1":
			df_corrected = correct_temperature_compensation(df.copy(deep=True), constant_1=constant, constant_2=other_constant)
		if target_spot == "BC2":
			df_corrected = correct_temperature_compensation(df.copy(deep=True), constant_2=constant, constant_1=other_constant)

		
		df_corrected.loc[df_corrected['Corrected IR ATN1'] <= 3, 'Corrected IR BCc'] = df_corrected.loc[df_corrected['Corrected IR ATN1'] <= 3, 'Corrected IR BC1']
		df_corrected.loc[df_corrected['Corrected IR ATN2'] <= 3, 'Corrected IR BCc'] = df_corrected.loc[df_corrected['Corrected IR ATN2'] <= 3, 'Corrected IR BC2']


		df_corrected = df_corrected.set_index('datetime_local')

		# Resampling the data
		numeric_columns = df_corrected.select_dtypes(include=['number']).columns
		non_numeric_cols = df_corrected.select_dtypes(exclude=['number']).columns
		
		df_resampled_numeric = df_corrected[numeric_columns].resample(resample_timebase).mean()
		df_resampled_non_numeric = df_corrected[non_numeric_cols].resample(resample_timebase).first()

		# Combine numeric and non-numeric columns
		df_resampled = pd.concat([df_resampled_numeric, df_resampled_non_numeric], axis=1).reset_index()

		# Calculate the RG value
		df_resampled[f'RG {target_spot}'] = 0.5 * (df_resampled[f'Green {target_spot}'] + df_resampled[f'Red {target_spot}'])

		# Calculate metrics for the current constant_1 value
		metrics = calculate_metrics(df_resampled, target_column, comparison_columns)

		# Append results
		result_entry = {
			'constant': constant,
			**{f'R^2 {col}': values['R^2'] for col, values in metrics.items()},
			**{f'RMSE {col}': values['RMSE'] for col, values in metrics.items()}
		}
		results.append(result_entry)

	# Create DataFrame from results
	results_df = pd.DataFrame(results)
	return results_df

## Problem solved here: trying to optimize a temperature compensation factor given unfiltered data from outdoor monitors
# solution is to minimize error between average of red and green signals and the IR signal
def optimize_correction_factor(df, target_spot='BC1', min = 0.02, max = 0.1, num = 15, other_constant = 0.018, resample_timebase = '15min', max_atn = 3):
	import numpy as np
	import pandas as pd
	target_column = f"RG {target_spot}"
	if target_spot == "BC1":
		comparison_columns = ['Corrected IR BC1', 'IR BC1']

	if target_spot == "BC2":
		comparison_columns = ['Corrected IR BC2', 'IR BC2']



	# Range of constant_1 values to iterate over
	constant_values = np.linspace(min, max, num)  # Example: 15 values from 0.02 to 0.1

	# DataFrame to store results
	results = []

	# Iterate over each value of constant_1
	for constant in constant_values:
		# Apply the temperature compensation function
		if target_spot == "BC1":
			df_corrected = correct_temperature_compensation(df.copy(deep=True), constant_1=constant, constant_2=other_constant, max = max_atn)
		if target_spot == "BC2":
			df_corrected = correct_temperature_compensation(df.copy(deep=True), constant_2=constant, constant_1=other_constant, max = max_atn)

		
		df_corrected.loc[df_corrected['Corrected IR ATN1'] <= 3, 'Corrected IR BCc'] = df_corrected.loc[df_corrected['Corrected IR ATN1'] <= 3, 'Corrected IR BC1']
		df_corrected.loc[df_corrected['Corrected IR ATN2'] <= 3, 'Corrected IR BCc'] = df_corrected.loc[df_corrected['Corrected IR ATN2'] <= 3, 'Corrected IR BC2']


		df_corrected = df_corrected.set_index('datetime_local')

		# Resampling the data
		numeric_columns = df_corrected.select_dtypes(include=['number']).columns
		non_numeric_cols = df_corrected.select_dtypes(exclude=['number']).columns
		
		df_resampled_numeric = df_corrected[numeric_columns].resample(resample_timebase).mean()
		df_resampled_non_numeric = df_corrected[non_numeric_cols].resample(resample_timebase).first()

		# Combine numeric and non-numeric columns
		df_resampled = pd.concat([df_resampled_numeric, df_resampled_non_numeric], axis=1).reset_index()

		# Calculate the RG value
		df_resampled[f'RG {target_spot}'] = 0.5 * (df_resampled[f'Green {target_spot}'] + df_resampled[f'Red {target_spot}'])

		# Calculate metrics for the current constant_1 value
		metrics = calculate_metrics(df_resampled, target_column, comparison_columns)

		# Append results
		result_entry = {
			'constant': constant,
			**{f'R^2 {col}': values['R^2'] for col, values in metrics.items()},
			**{f'RMSE {col}': values['RMSE'] for col, values in metrics.items()}
		}
		results.append(result_entry)

	# Create DataFrame from results
	results_df = pd.DataFrame(results)
	return results_df


# correct dATN with leak compensation
def correct_with_leak_compensation(calibration_df, leak_correction_df, name = "", r_squared_threshold = 0.2):
	# Merge calibration_df with results_df to get correction factors

	leak_correction_df.loc[leak_correction_df['R^2'] <= r_squared_threshold, 'Slope'] = 1

	calibration_df = calibration_df.merge(leak_correction_df[['Serial number', 'test', 'Tape position', 'Slope']],
										on=['Serial number', 'test', 'Tape position'],
										how='left')

	# Apply the correction factor
	for wl in ['IR', 'UV', 'Blue', 'Red', 'Green']:
		col_original = f'delta {wl} ATN2'
		col_corrected = name + f'delta {wl} ATN2'
		# Calculate corrected values
		calibration_df[col_corrected] = calibration_df[col_original] / calibration_df['Slope']

	return calibration_df

# correct BC for a reference device that is leak compensated 
# because need to recalcualte bC with leak compensation for error calculations
def correct_BC_reference(df):
	import pandas as pd
	df_calibration_corrected = perform_first_principle_calcs_corrected(#perform_first_principle_calcs_sanity(
		df,
		wl="IR",
		dev_type="MA350",
		flow_var1='Flow1 (mL/min)',
		flow_var2='Flow2 (mL/min)',
		al30_sigma = 12.5*1,
		al60_sigma = 12.5*1,
		al80_sigma = 12.5*1,
		delta_time_var = 'delta_time'
	)

	for wl in ['Blue', 'Red', 'Green', 'UV']:
		next_df = perform_first_principle_calcs_corrected(
			df_calibration_corrected,
			wl=wl,
			dev_type="MA350",
			flow_var1='Flow1 (mL/min)',
			flow_var2='Flow2 (mL/min)',
			al30_sigma = 12.5*1,
			al60_sigma = 12.5*1,
			al80_sigma = 12.5*1,
			delta_time_var = 'delta_time'
		)

		smaller_df = next_df[['datetime_local', 'Serial number', f'Corrected {wl} BCc', f'Corrected {wl} BC1', f'Corrected {wl} BC2']].reset_index(drop = True)
		df_calibration_corrected = pd.merge(df_calibration_corrected, smaller_df, on=['Serial number', 'datetime_local'], how='inner')	

	df_calibration_corrected = df_calibration_corrected.reset_index(drop = True)
	return df_calibration_corrected

# Correct ATN for spot2 for leak compensation validations
def correct_ATN_leak_compensation(df_corrected):
	# Function to apply the cumulative sum logic per device
	def calculate_corrected_atn(group):
		#print(group['Serial number'].unique())
		# Assuming wavelengths are defined like this
		wavelengths = ['UV2', 'Blue2', 'Green2', 'Red2', 'IR2']

		vars = []
		for wl in wavelengths:
			var = f'{wl[:-1]} ATN{wl[-1]}'
			vars.append(var)

		for var in vars:
			# Initialize the corrected ATN with the original ATN values
			group[f'Corrected {var}'] = group[f'{var}']
			
			# Iterate through the group's rows
			for i in range(1, len(group)):
				if group.iloc[i]['Tape position'] != group.iloc[i-1]['Tape position'] or \
					(group.iloc[i]['datetime_local'] - group.iloc[i-1]['datetime_local']).total_seconds() > (int(group.iloc[i]['Timebase (s)']) * 5):
					# If the time gap is greater than 2 minutes, reset the cumulative sum
					group.at[group.index[i], f'Corrected {var}'] = group.iloc[i][f'{var}']

				# if large jump, find the two uncorrected ATN values, multiply the diff by the calibration factor, then add to corrected ATN to get next datapoint
				elif (group.iloc[i]['datetime_local'] - group.iloc[i-1]['datetime_local']).total_seconds() > (int(group.iloc[i]['Timebase (s)']) * 1.5):
					group.at[group.index[i], f'Corrected {var}'] = (group.iloc[i][f'{var}'] - group.iloc[i - 1][f'{var}']) * group.iloc[i][f'Slope'] + group.iloc[i-1][f'Corrected {var}']
				else:
					# Otherwise, add corrected delta ATN to corrected ATN at each timestep
					group.at[group.index[i], f'Corrected {var}'] = group.iloc[i - 1][f'Corrected {var}'] + group.iloc[i][f'Corrected delta {var}']
				
		return group

	# Apply the function to each device group
	df_corrected = df_corrected.groupby('Serial number', group_keys=False).apply(calculate_corrected_atn)

	for wl in ['IR', "UV", "Green", "Red", "Blue"]:
		df_corrected[f"Corrected {wl} ATN1"] = df_corrected[f"{wl} ATN1"]
		df_corrected[f"Corrected delta {wl} ATN1"] = df_corrected[f"delta {wl} ATN1"]

	return df_corrected


def calculate_metrics(df, target_column, compare_columns):
	from sklearn.metrics import r2_score, mean_squared_error
	import numpy as np
	import pandas as pd
	"""
	Calculate R^2 and RMSE for specified comparison columns against a target column.

	Parameters:
	df (pd.DataFrame): DataFrame containing the data.
	target_column (str): The name of the target column.
	compare_columns (list): List of comparison column names.

	Returns:
	dict: A dictionary containing R^2 and RMSE for each comparison column.
	"""
	# Make a copy of the input DataFrame
	df_copy = df.copy()

	# Drop rows with any NaN values in the selected comparison columns
	df_copy = df_copy.dropna(subset=[target_column] + compare_columns)
	df_copy = df_copy.loc[df_copy[target_column] < 500_000]

	# Dictionary to store results
	results = {}

	# Loop through the comparison columns to calculate R^2 and RMSE
	for col in compare_columns:
		# Calculate R^2 and RMSE
		df_copy = df_copy.loc[df_copy[col] < 500_000]

		r2 = r2_score(df_copy[target_column], df_copy[col])
		rmse = np.sqrt(mean_squared_error(df_copy[target_column], df_copy[col]))
		
		# Store results
		results[col] = {'R^2': round(r2,3), 'RMSE': round(rmse,3)}

	return results


def plot_corrected_ir_bc(df, target_spot='BC1', constant_1=0.09, constant_2=0.018, max_atn = 1):
	"""
	Plot Corrected IR values, IR values, and the average of Red and Green values over time.

	Parameters:
	df (pd.DataFrame): The input DataFrame containing the required columns.
	target_spot (str): The spot to choose ('1', '2', or 'c') for plotting.
	constant_1 (float): The value for constant_1 in temperature compensation.
	constant_2 (float): The value for constant_2 in temperature compensation.
	"""
	import pandas as pd
	import plotly.graph_objects as go

	# Map target_spot to actual column names
	spot_mapping = {
		'1': 'BC1',
		'2': 'BC2',
		'c': 'BCc'
	}

	if target_spot not in spot_mapping:
		raise ValueError("Invalid target_spot. Choose '1', '2', or 'c'.")

	target_spot_name = spot_mapping[target_spot]

	# Apply the temperature compensation
	df_corrected = correct_temperature_compensation(df.copy(deep=True), 
																constant_1=constant_1, 
																constant_2=constant_2,
																max = max_atn)

	# Adjust Corrected IR BCc based on the condition
	df_corrected.loc[df_corrected['Corrected IR ATN1'] <= 3, f'Corrected IR {target_spot_name}'] = df_corrected.loc[df_corrected['Corrected IR ATN1'] <= 3, f'Corrected IR {target_spot_name}']

	# Set the datetime index
	df_corrected = df_corrected.set_index('datetime_local')

	# Resample the data
	numeric_columns = df_corrected.select_dtypes(include=['number']).columns
	non_numeric_cols = df_corrected.select_dtypes(exclude=['number']).columns
	df_resampled_numeric = df_corrected[numeric_columns].resample("15min").mean()
	df_resampled_non_numeric = df_corrected[non_numeric_cols].resample("15min").first()

	# Combine numeric and non-numeric columns
	df_resampled = pd.concat([df_resampled_numeric, df_resampled_non_numeric], axis=1)
	df_resampled = df_resampled.reset_index()

	# Calculate RG values based on the selected target spot
	df_resampled[f'RG {target_spot_name}'] = 0.5 * (df_resampled[f'Green {target_spot_name}'] + df_resampled[f'Red {target_spot_name}'])

	# Initialize the figure
	fig = go.Figure()

	# Add trace for Corrected IR for the selected spot
	fig.add_trace(go.Scatter(
		x=df_resampled['datetime_local'],
		y=df_resampled[f'Corrected IR {target_spot_name}'],
		mode='lines',
		name=f'Corrected IR {target_spot_name}',
		line=dict(color='blue')
	))

	# Add trace for IR for the selected spot
	fig.add_trace(go.Scatter(
		x=df_resampled['datetime_local'],
		y=df_resampled[f'IR {target_spot_name}'],
		mode='lines',
		name=f'IR {target_spot_name}',
		line=dict(color='red')
	))

	# Add trace for RG for the selected spot
	fig.add_trace(go.Scatter(
		x=df_resampled['datetime_local'],
		y=df_resampled[f'Red {target_spot_name}'],
		mode='lines',
		name=f'Red {target_spot_name}',
		line=dict(color='red')
	))

		# Add trace for RG for the selected spot
	fig.add_trace(go.Scatter(
		x=df_resampled['datetime_local'],
		y=df_resampled[f'Green {target_spot_name}'],
		mode='lines',
		name=f'Green {target_spot_name}',
		line=dict(color='green')
	))

	# Add trace for RG for the selected spot
	fig.add_trace(go.Scatter(
		x=df_resampled['datetime_local'],
		y=df_resampled[f'UV {target_spot_name}'],
		mode='lines',
		name=f'UV {target_spot_name}',
		line=dict(color='purple')
	))

	# Update layout
	fig.update_layout(
		title=f'Corrected IR {target_spot_name}, IR {target_spot_name}, and avg of Red and Green {target_spot_name} Over Time',
		xaxis_title='Time',
		yaxis_title='Concentration (ng/m³)',
		legend=dict(
			x=0,
			y=1,
			traceorder='normal',
			font=dict(size=12),
			bordercolor='Black',
			borderwidth=1
		)
	)

	# Show the plot
	return fig, df_corrected, df_resampled


# generate leak compensation flow factors fro spot 2 for each tape position under IR ATN 3
def generate_tape_specific_atn_flow_factors(calibration_df, IR = False):
	from sklearn.linear_model import LinearRegression
	from sklearn.metrics import r2_score
	import pandas as pd

	calibration_df = calibration_df.loc[calibration_df['IR ATN1'] < 3]
	results = []
	for (serial, test, tape_position), group in calibration_df.groupby(['Serial number', 'test', 'Tape position']):
		slopes = []
		rs = []
		for wl in ['IR', 'UV', 'Blue', 'Red', "Green"]:

			x = group[f"delta {wl} ATN1"] / group['Flow1 (mL/min)']
			y = group[f'delta {wl} ATN2'] / group['Flow2 (mL/min)']
			
			# Reshape x for sklearn
			valid_idx = ~x.isna() & ~y.isna()  # Find indices where neither x nor y is NaN
			x = x[valid_idx].values.reshape(-1, 1)  # Reshape x for sklearn
			y = y[valid_idx].values
			
			# Perform linear regression with intercept forced to zero
			model = LinearRegression(fit_intercept=False)  # Force intercept to zero
			model.fit(x, y)
			
			# Get the slope
			slope = model.coef_[0]
			
			# Predict y using the model
			y_pred = model.predict(x)
			
			# Calculate R^2
			r_squared = r2_score(y, y_pred)
			#print(wl, serial, r_squared)
			slopes.append(slope)
			rs.append(r_squared)
			# Save the results
		
		if IR:
			results.append({
				'Serial number': serial,
				'test': test,
				'Tape position': tape_position,
				f'Slope': slopes[0],
				f'R^2': rs[0],
				'length': len(group.index)
			})
		else:
			results.append({
				'Serial number': serial,
				'test': test,
				'Tape position': tape_position,
				f'Slope': np.mean(slopes),
				f'R^2': np.mean(rs),
				'length': len(group.index)
			})
		# results.append({
		#     'Serial number': serial,
		#     'test': test,
		#     'Tape position': tape_position,
		#     f'IR Slope': slopes[0],
		#     f'IR R^2': rs[0],
		#     f'UV Slope': slopes[1],
		#     f'UV R^2': rs[1],
		#     f'Blue Slope': slopes[2],
		#     f'Blue R^2': rs[2],
		#     f'Red Slope': slopes[3],
		#     f'Red R^2': rs[3],
		#     f'Green Slope': slopes[4],
		#     f'Green R^2': rs[4]})
			

	# Convert results to a DataFrame
	results_df = pd.DataFrame(results)
	return results_df


def read_AE33(AE33_dir):
	import pandas as pd
	import os
	from pathlib import Path

	import glob
	#dat_files = glob.glob(os.path.join(AE33_dir, '*.dat')) 
	dat_files = list(Path(AE33_dir).rglob('*.dat'))


	# Define the mapping of old column names to new names
	column_mapping = {
		'Date(yyyy/MM/dd)': 'date',
		'Time(hh:mm:ss)': 'time',
		'Timebase': 'Timebase (s)',
		'RefCh1': 'Channel 1 Ref',
		'Sen1Ch1': 'Channel 1 Sen1',
		'Sen2Ch1': 'Channel 1 Sen2',
		'RefCh2': 'Channel 2 Ref',
		'Sen1Ch2': 'Channel 2 Sen1',
		'Sen2Ch2': 'Channel 2 Sen2',
		'RefCh3': 'Channel 3 Ref',
		'Sen1Ch3': 'Channel 3 Sen1',
		'Sen2Ch3': 'Channel 3 Sen2',
		'RefCh4': 'Channel 4 Ref',
		'Sen1Ch4': 'Channel 4 Sen1',
		'Sen2Ch4': 'Channel 4 Sen2',
		'RefCh5': 'Channel 5 Ref',
		'Sen1Ch5': 'Channel 5 Sen1',
		'Sen2Ch5': 'Channel 5 Sen2',
		'RefCh6': 'Channel 6 Ref',
		'Sen1Ch6': 'Channel 6 Sen1',
		'Sen2Ch6': 'Channel 6 Sen2',
		'RefCh7': 'Channel 7 Ref',
		'Sen1Ch7': 'Channel 7 Sen1',
		'Sen2Ch7': 'Channel 7 Sen2',
		'Flow1': 'Flow1 (mL/min)',
		'Flow2': 'Flow2 (mL/min)',
		'FlowC': 'Flow total (mL/min)',
		'Pressure(Pa)': 'Internal pressure (Pa)',
		'Temperature(°C)': 'Internal temp (C)',
		'BB(%)': 'BB (%)',
		'SupplyTemp': 'Sample temp (C)',
	}

	column_names = [
		'Date(yyyy/MM/dd)', 'Time(hh:mm:ss)', 'Timebase', 'RefCh1', 'Sen1Ch1', 'Sen2Ch1', 
		'RefCh2', 'Sen1Ch2', 'Sen2Ch2', 'RefCh3', 'Sen1Ch3', 'Sen2Ch3', 'RefCh4', 
		'Sen1Ch4', 'Sen2Ch4', 'RefCh5', 'Sen1Ch5', 'Sen2Ch5', 'RefCh6', 'Sen1Ch6', 
		'Sen2Ch6', 'RefCh7', 'Sen1Ch7', 'Sen2Ch7', 'Flow1', 'Flow2', 'FlowC', 
		'Pressure(Pa)', 'Temperature(°C)', 'BB(%)', 'ContTemp', 'SupplyTemp', 'Status', 
		'ContStatus', 'DetectStatus', 'LedStatus', 'ValveStatus', 'LedTemp', 'BC11', 
		'BC12', 'BC1', 'BC21', 'BC22', 'BC2', 'BC31', 'BC32', 'BC3', 'BC41', 
		'BC42', 'BC4', 'BC51', 'BC52', 'BC5', 'BC61', 'BC62', 'BC6', 'BC71', 
		'BC72', 'BC7', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'TapeAdvCount'
	]

	files = []
	for file in dat_files:
		print(file)
		file = str(file)
		if "AE3s3_log_AE33" not in file:
			try:
				df = pd.read_csv(file, delimiter=' ', header=None, skiprows=8)
				df.columns = column_names + list(df.columns[len(column_names):])
				files.append(df)
			except Exception as e:
				print(f"Error reading {file}: {e}")
				continue

	AE33_df = pd.concat(files)

	AE33_df = AE33_df.rename(columns = column_mapping)
	AE33_df['IR BCc'] = AE33_df['BC6']
	AE33_df['IR BC1'] = AE33_df['BC61']
	AE33_df['IR BC2'] = AE33_df['BC62']
	AE33_df['datetime_local'] = pd.to_datetime(AE33_df['date'] + ' ' + AE33_df['time'], format='%Y/%m/%d %H:%M:%S')

	#AE33_df['IR BC1'] = AE33_df['BC'] * 1000
	return AE33_df




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
def plot_seasonal_diurnal_profiles_multi_df(dfs, df_labels, column_name, timebase='datetime_local', agg_minutes=15, colors=None):
    """
    Plots seasonal diurnal profiles with 95% confidence intervals for one column
    across multiple DataFrames (e.g., different sites).

    Args:
        dfs (list of pd.DataFrame): List of DataFrames.
        df_labels (list of str): Labels for each DataFrame (e.g., site names).
        column_name (str): Name of the column to plot.
        timebase (str): Datetime column name.
        agg_minutes (int): Time bin size in minutes.
        colors (list of str): Optional list of colors for each DataFrame.
    """
    if colors is None:
        colors = plt.cm.tab10.colors[:len(dfs)]

    day_types = ['Working Day', 'Saturday', 'Sunday']
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(24, 16), sharex=True, sharey=True)

    for idx, (df, label) in enumerate(zip(dfs, df_labels)):
        df = df.copy()
        df[timebase] = pd.to_datetime(df[timebase])
        df['month'] = df[timebase].dt.month
        df['weekday'] = df[timebase].dt.weekday
        df['day_type'] = np.where(df['weekday'] < 5, 'Working Day',
                                  np.where(df['weekday'] == 5, 'Saturday', 'Sunday'))

        # Map months to seasons
        month_to_season = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        df['season'] = df['month'].map(month_to_season)

        # Create time bins
        minutes_since_midnight = df[timebase].dt.hour * 60 + df[timebase].dt.minute
        df['time_bin'] = (minutes_since_midnight // agg_minutes).astype(int)

        grouped = df.groupby(['season', 'day_type', 'time_bin']).agg(
            mean_value=(column_name, 'mean'),
            std_value=(column_name, 'std'),
            count_value=(column_name, 'count')
        ).reset_index()

        # Calculate 95% CI
        alpha = 0.05
        grouped['t_value'] = stats.t.ppf(1 - alpha / 2, grouped['count_value'] - 1)
        grouped['ci_lower'] = grouped['mean_value'] - grouped['t_value'] * (grouped['std_value'] / np.sqrt(grouped['count_value']))
        grouped['ci_upper'] = grouped['mean_value'] + grouped['t_value'] * (grouped['std_value'] / np.sqrt(grouped['count_value']))

        for i, day in enumerate(day_types):
            for j, season in enumerate(seasons):
                ax = axes[i, j]
                data = grouped[(grouped['day_type'] == day) & (grouped['season'] == season)]
                x_hours = data['time_bin'] * agg_minutes / 60.0

                ax.plot(x_hours, data['mean_value'], label=label, color=colors[idx], linewidth=2)
                ax.fill_between(x_hours, data['ci_lower'], data['ci_upper'], color=colors[idx], alpha=0.2)

                if i == 0:
                    ax.set_title(season, fontsize=25)

                # Weekday label on the left
                if j == 0:
                    ax.set_ylabel(day, fontsize=25, loc='center')

                # Y-axis label with units on the right
                if j == len(seasons) - 1:
                    ax.yaxis.set_label_position('right')   # Move label to right
                    ax.yaxis.tick_right()                  # Show ticks on right side
                    ax.set_ylabel(f"{column_name} (ng/m³)", fontsize=25, labelpad = 30)
                    ax.yaxis.set_ticks_position('right')  # Ensure ticks on right side only
                else:
                    # Remove left ticks but keep grid lines by hiding left ticks & labels
                    ax.yaxis.set_ticks_position('none')   # Disable ticks on both sides
                    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)


                # X-axis label on bottom row
                if i == 2:
                    ax.set_xlabel('Hour of Day', fontsize=25)

                ax.tick_params(labelsize=22)
                ax.set_xlim(0, 24)
                ax.grid(True)

    # Set global title
    combined_label = " vs. ".join(df_labels)
    fig.suptitle(f"Seasonal Diurnal Profiles of {column_name} at {combined_label}", fontsize=40, y=1)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(df_labels), fontsize=30,)
    bbox_to_anchor=(0.5, 0.05),  # x=0.5 center, y=-0.05 below figure

    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_seasonal_diurnal_profiles_multi_column(df, column_names, timebase='datetime_local', agg_minutes=15, colors=None):
    """
    Plots seasonal diurnal profiles with 95% confidence intervals for multiple columns,
    separated by Working Days, Saturday, and Sunday.

    Args:
        df (pd.DataFrame): Input DataFrame with datetime and measurement columns.
        column_names (list of str): List of column names to plot.
        timebase (str): Datetime column name.
        agg_minutes (int): Time bin size in minutes.
        colors (list of str): Optional list of colors for each column.
    """
    if colors is None:
        colors = plt.cm.tab10.colors[:len(column_names)]

    df = df.copy()
    df[timebase] = pd.to_datetime(df[timebase])
    df['month'] = df[timebase].dt.month
    df['weekday'] = df[timebase].dt.weekday
    df['day_type'] = np.where(df['weekday'] < 5, 'Working Day',
                              np.where(df['weekday'] == 5, 'Saturday', 'Sunday'))
    
    # Define seasons
    month_to_season = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    df['season'] = df['month'].map(month_to_season)

    # Create time bins
    minutes_since_midnight = df[timebase].dt.hour * 60 + df[timebase].dt.minute
    df['time_bin'] = (minutes_since_midnight // agg_minutes).astype(int)

    day_types = ['Working Day', 'Saturday', 'Sunday']
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 12), sharex=True, sharey=True)

    for col_idx, column_name in enumerate(column_names):
        grouped = df.groupby(['season', 'day_type', 'time_bin']).agg(
            mean_value=(column_name, 'mean'),
            std_value=(column_name, 'std'),
            count_value=(column_name, 'count')
        ).reset_index()

        alpha = 0.05
        grouped['t_value'] = stats.t.ppf(1 - alpha / 2, grouped['count_value'] - 1)
        grouped['ci_lower'] = grouped['mean_value'] - grouped['t_value'] * (grouped['std_value'] / np.sqrt(grouped['count_value']))
        grouped['ci_upper'] = grouped['mean_value'] + grouped['t_value'] * (grouped['std_value'] / np.sqrt(grouped['count_value']))

        for i, day in enumerate(day_types):
            for j, season in enumerate(seasons):
                ax = axes[i, j]
                data = grouped[(grouped['day_type'] == day) & (grouped['season'] == season)]
                x_hours = data['time_bin'] * agg_minutes / 60.0

                ax.plot(x_hours, data['mean_value'], label=column_name, color=colors[col_idx])
                ax.fill_between(x_hours, data['ci_lower'], data['ci_upper'], color=colors[col_idx], alpha=0.2)

                if i == 0:
                    ax.set_title(season)
                if j == 0:
                    ax.set_ylabel(day)
                ax.set_xlim(0, 24)
                ax.grid(True)

    for ax in axes.flatten():
        ax.set_xlabel('Hour of Day')
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_monthly_diurnal_profile_with_ci_multi(dfs, column_name, timebase='datetime_local', agg_minutes=15, labels=None, colors=None):
    """
    Plots monthly diurnal profiles with 95% confidence intervals for multiple DataFrames
    on the same figure, using user-defined aggregation intervals.

    Args:
        dfs (list of pd.DataFrame): List of input DataFrames.
        column_name (str): The column for which to plot the diurnal profile.
        timebase (str): Name of the datetime column (default is 'datetime_local').
        agg_minutes (int): Aggregation period in minutes (default is 15).
        labels (list of str): Optional list of labels for each DataFrame.
        colors (list of str): Optional list of colors for each DataFrame.

    Returns:
        None
    """
    if labels is None:
        labels = [f'Data {i+1}' for i in range(len(dfs))]
    if colors is None:
        colors = plt.cm.tab10.colors[:len(dfs)]

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 10))
    axes = axes.flatten()

    for df_idx, df in enumerate(dfs):
        df = df.copy()
        df[timebase] = pd.to_datetime(df[timebase])
        df['month'] = df[timebase].dt.month
        minutes_since_midnight = df[timebase].dt.hour * 60 + df[timebase].dt.minute
        df['time_bin'] = (minutes_since_midnight // agg_minutes).astype(int)

        grouped = df.groupby(['month', 'time_bin']).agg(
            mean_value=(column_name, 'mean'),
            std_value=(column_name, 'std'),
            count_value=(column_name, 'count')
        ).reset_index()

        alpha = 0.05
        t_value = stats.t.ppf(1 - alpha / 2, grouped['count_value'] - 1)
        grouped['ci_lower'] = grouped['mean_value'] - t_value * (grouped['std_value'] / np.sqrt(grouped['count_value']))
        grouped['ci_upper'] = grouped['mean_value'] + t_value * (grouped['std_value'] / np.sqrt(grouped['count_value']))

        for month in range(1, 13):
            ax = axes[month - 1]
            month_data = grouped[grouped['month'] == month]
            x_hours = month_data['time_bin'] * agg_minutes / 60.0

            ax.plot(x_hours, month_data['mean_value'], label=labels[df_idx], color=colors[df_idx])
            ax.fill_between(x_hours, month_data['ci_lower'], month_data['ci_upper'], color=colors[df_idx], alpha=0.2)

            ax.set_title(f'Month {month}')
            ax.set_xlim(0, 24)
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel(column_name)
            ax.grid(True)

    for ax in axes:
        ax.legend()

    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import calendar
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_monthly_diurnal_profiles_multi_column(df, column_names, timebase='datetime_local', agg_minutes=15, colors=None, ylabel='Value', title=None, save_path=False):
    """
    Plots monthly diurnal profiles with 90% confidence intervals for multiple columns
    from a single DataFrame, all on the same figure with monthly subplots.

    Args:
        df (pd.DataFrame): Input DataFrame with time-based and numeric data.
        column_names (list of str): List of column names to plot.
        timebase (str): Name of the datetime column (default is 'datetime_local').
        agg_minutes (int): Aggregation period in minutes (default is 15).
        colors (list of str): Optional list of colors for each column.
        ylabel (str): Label for the y-axis.
        title (str): Overall figure title.
        save_path (str or False): File path to save the figure. If False, shows the plot.

    Returns:
        None
    """
    if colors is None:
        colors = plt.cm.tab10.colors[:len(column_names)]

    df = df.copy()
    df[timebase] = pd.to_datetime(df[timebase])
    df['month'] = df[timebase].dt.month
    minutes_since_midnight = df[timebase].dt.hour * 60 + df[timebase].dt.minute
    df['time_bin'] = (minutes_since_midnight // agg_minutes).astype(int)

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 10))
    axes = axes.flatten()

    # Precompute global CI limits
    ci_lowers = []
    ci_uppers = []

    for col_idx, column_name in enumerate(column_names):
        grouped = df.groupby(['month', 'time_bin']).agg(
            mean_value=(column_name, 'mean'),
            std_value=(column_name, 'std'),
            count_value=(column_name, 'count')
        ).reset_index()

        alpha = 0.10  # 90% CI
        t_value = stats.t.ppf(1 - alpha / 2, grouped['count_value'] - 1)
        grouped['ci_lower'] = grouped['mean_value'] - t_value * (grouped['std_value'] / np.sqrt(grouped['count_value']))
        grouped['ci_upper'] = grouped['mean_value'] + t_value * (grouped['std_value'] / np.sqrt(grouped['count_value']))

        ci_lowers.append(grouped['ci_lower'].min())
        ci_uppers.append(grouped['ci_upper'].max())

        for month in range(1, 13):
            ax = axes[month - 1]
            month_data = grouped[grouped['month'] == month]
            x_hours = month_data['time_bin'] * agg_minutes / 60.0

            ax.plot(x_hours, month_data['mean_value'], label=column_name, color=colors[col_idx])
            ax.fill_between(x_hours, month_data['ci_lower'], month_data['ci_upper'], color=colors[col_idx], alpha=0.2)

            ax.set_title(calendar.month_name[month], fontsize=20)
            ax.set_xlim(0, 24)
            ax.set_xlabel('Hour of Day', fontsize=16)
            ax.set_ylabel(ylabel, fontsize=16)
            ax.tick_params(axis='both', labelsize=16)
            ax.grid(True)

    # Set global y-axis limits
    ymin = min(ci_lowers)
    ymax = max(ci_uppers)
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    # Single legend for figure
    handles = [
        Line2D([0], [0], color=colors[col_idx], lw=2, label='Mean Diurnal Profile'),
        Patch(facecolor=colors[col_idx], alpha=0.2, label='90% Confidence Interval')
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=20)

    if title:
        fig.suptitle(title, fontsize=24)

    plt.tight_layout(rect=[0, 0, 1, 0.975])  # Leave room for title and legend

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
    else:
        plt.show()

def time_series_multi(
    df, 
    x_col='datetime_local', 
    y_cols=['AAE_BrC'], 
    title='Time Series Plot',
    colors=None,
    bottom_zero = True,
):
    df = df.copy()
    try:
        df[x_col] = pd.to_datetime(df[x_col])
    except Exception as e:
        print(f"Error converting {x_col} to datetime: {e}")

    if colors is None:
        colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(16, 4))

    max_val = 0
    min_val = 0

    for i, y_col in enumerate(y_cols):
        if y_col in df.columns:
            ax.plot(
                df[x_col], df[y_col], 
                label=y_col, 
                color=colors[i % len(colors)],
                linewidth=2.5,
                alpha=0.9
            )
            max_val  = max(max_val, df[y_col].max(skipna=True))
            min_val = min(min_val, df[y_col].quantile(0.01, interpolation='nearest'))


    # Axis formatting
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date and Time', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    # Set y-limits
    if bottom_zero: min_val = 0
    ax.set_ylim(min_val, max_val * 1.05)

    # Improve layout and appearance
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=12)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


### From aethpy.dataparse, used to create a list of tape positions in the master df
def create_df_list_atlevel_sid(df):
    df_interim_list = []

    for sn in df['Serial number'].unique():
        df_i1 = df.loc[df['Serial number'] == sn, :]

        for sid in df_i1['Session ID'].unique():
            dfi2 = df_i1.loc[df_i1['Session ID'] == sid]

            df_interim_list.append(dfi2)##

    return df_interim_list

from re import compile as re_compile
regex_alx = re_compile(r'AL\d+')

def create_df_list_atlevel_tapeposition(
        df
    ):
	df_interim_list = []

	for df_device in create_df_list_atlevel_sid(df):
		if ('AE' in df_device['Serial number'].unique()[0]) or (regex_alx.search(df_device['Serial number'].unique()[0])):
			df_device['Tape position'] = '-1'
			df_interim_list.append(df_device)  
		elif 'PAX' in df_device['Serial number'].unique()[0]:
			df_interim_list.append(df_device)       
			df_device['Tape position'] = '-1'
		else:
			for tp in df_device['Tape position'].unique():
				df_interim = df_device.loc[df_device['Tape position'] == tp, :]
				# df_interim = df_device.loc[df_device['Tape position'] == tp, :].reset_index(drop=True)
				df_interim_list.append(df_interim)

	return df_interim_list


def relative_tape_position_calc(
        dataframe,
        bytest=False,
        use_API_varnames=False,
    ):
    df = dataframe.copy(deep=True)

    # set variable names
    if not use_API_varnames:
        sn_var = 'Serial number'
        datetime_var = 'datetime_local'
        tp_var = 'Tape position'
    else:
        sn_var = 'deviceId'
        datetime_var = 'unixtime'
        tp_var = 'sessionId'

    snlist = df[sn_var].unique()

    df_list = []
    for sn in snlist:
        df_i =\
            df.loc[df[sn_var]==sn,:].sort_values(by=datetime_var,ascending=True).reset_index(drop=True)
        if ('AE51' in sn) or regex_alx.search(sn) or ('PAX' in sn):
            df_i[tp_var] = 0
            df_i['Relative tape position'] = 1
            df_list.append(df_i)
        else:
            # relative by test
            if bytest:
                for test in df_i['test'].unique():
                    df_j = df_i.loc[df_i['test']==test]
                    # assess when a tape position has changed
                    df_j['Tape position shifted'] =\
                        df_j[tp_var].shift(1)
                    df_j.loc[~df_j['Tape position shifted'].isna(), 'Tape position delta'] =\
                        df_j.loc[~df_j['Tape position shifted'].isna(), tp_var].astype(float) -\
                        df_j.loc[~df_j['Tape position shifted'].isna(), 'Tape position shifted'].astype(float)
                    # for any tape position delta that is not 0, make tape position delta 1 to allow cumsum incrementing
                    df_j['Tape position delta boolean'] = 0
                    df_j.loc[
                            (df_j['Tape position delta'] > 0) | 
                            (df_j['Tape position delta'] < 0),
                        'Tape position delta boolean'
                        ] = 1
                    df_j['Relative tape position'] =\
                        df_j['Tape position delta boolean'].cumsum(skipna=True) + 1
                    df_list.append(df_j)
            # relative overall
            else:
                # assess when a tape position has changed
                df_i['Tape position shifted'] =\
                    df_i[tp_var].shift(1)
                df_i.loc[~df_i['Tape position shifted'].isna(), 'Tape position delta'] =\
                    df_i.loc[~df_i['Tape position shifted'].isna(), tp_var].astype(float) -\
                    df_i.loc[~df_i['Tape position shifted'].isna(), 'Tape position shifted'].astype(float)
                # for any tape position delta that is not 0, make tape position delta 1 to allow cumsum incrementing
                df_i['Tape position delta boolean'] = 0
                df_i.loc[
                        (df_i['Tape position delta'] > 0) | 
                        (df_i['Tape position delta'] < 0),
                    'Tape position delta boolean'
                    ] = 1
                df_i['Relative tape position'] =\
                    df_i['Tape position delta boolean'].cumsum(skipna=True) + 1
                df_list.append(df_i)
    df_out = pd.arraysconcat(df_list).reset_index(drop=True)                

    return df_out


magic_integer_math_number = 65536

def ema_smoother_FW_filter(
        alpha: float,
        new_sample,
        ema_previous,
        integer_math = True
    ):

    if (new_sample == None) or (pd.isna(new_sample)) or (pd.isna(alpha)):
        ema_out = np.nan
    else:
        if integer_math:
            ### Fixed point math
            alpha_int =\
                (alpha*magic_integer_math_number)
            tmp =\
                int(new_sample) * ( alpha_int ) + int(ema_previous) * ( magic_integer_math_number - alpha_int )
            # update for fixed point math
            ema_out =\
                int((tmp + (magic_integer_math_number/2)) / magic_integer_math_number)
        else:
            ### float math
            ema_out = ( ( 1.0 - alpha ) * (ema_previous) + (alpha) * new_sample )
 
    return ema_out


def dema_smoother_FW_filter(
        i: int,
        alpha: float,
        new_sample: float,
        ema_previous, 
        ema_ema_previous,
        integer_math = True
    ):

    if pd.isna(new_sample):
        ema = np.nan
        ema_ema = np.nan
        dema = np.nan
        dema_for_out = np.nan
        # set for output / next sample
        ema_previous = np.nan
        ema_ema_previous = np.nan
    elif (i == 0) or (pd.isna(ema_previous)):
        ### Create EMA
        ema =\
            ema_smoother_FW_filter(
                alpha = alpha,
                new_sample = new_sample,
                ema_previous = new_sample,
                integer_math = integer_math
                )
        ## Create DEMA
        # EMA of EMA
        ema_ema =\
            ema_smoother_FW_filter(
                alpha = alpha,
                new_sample = ema,
                ema_previous = new_sample,
                integer_math = integer_math
            )
        # DEMA
        dema = (2 * ema) - ema_ema
        dema_for_out = dema
        # set for output / next sample
        ema_previous = ema
        ema_ema_previous = new_sample
    elif pd.isna(ema_ema_previous):
        ema =\
            ema_smoother_FW_filter(
                alpha = alpha,
                new_sample = new_sample,
                ema_previous = ema_previous,
                integer_math = integer_math
            )
        ema_ema =\
            ema_smoother_FW_filter(
                alpha = alpha,
                new_sample = ema,
                ema_previous = new_sample, # should this be ema_previous instead?
                integer_math = integer_math
                )
        # DEMA
        dema = (2 * ema) - ema_ema
        dema_for_out = dema
        # set for output / next sample
        ema_previous = ema
        ema_ema_previous = ema_ema
    else:
        ## create EMA
        ema =\
            ema_smoother_FW_filter(
                alpha = alpha,
                new_sample = new_sample,
                ema_previous = ema_previous,
                integer_math = integer_math
                )
        ## Create DEMA
        # EMA of EMA
        ema_ema =\
            ema_smoother_FW_filter(
                alpha = alpha,
                new_sample = ema,
                ema_previous = ema_previous,
                integer_math = integer_math
                )
        # DEMA
        dema = (2 * ema) - ema_ema
        dema_for_out = dema
        # set for output / next sample
        ema_previous = ema
        ema_ema_previous = ema_ema

    return ema, ema_ema, dema_for_out, ema_previous, ema_ema_previous



def dema_smoother_FW_implementation(
        dataframe,
        variable,
        output_var_name,
        alpha: float,
        auto_calc_alpha = True,
        smoothing_window_size_minutes = 15,
        integer_math = True,
        dev = 0,
        sn_var = 'Serial number',
        sid_var = 'Session ID',
        tp_var = 'Tape position',
        datetime_var = 'datetime_local'
    ):

    # if dev:
    #     ### for dev
    #     dataframe = df_j_i.copy(deep=True)
    #     variable = 'IR BCc'
    #     output_var_name = 'IR BCc smoothed'
    #     alpha = 0.125
    #     integer_math = True
    #     ###

    # sort
    dataframe.sort_values(by =[sn_var,sid_var,tp_var,datetime_var],ascending=True, inplace=True, ignore_index=True)

    ### apply by SID and TP
    df_i_list = []
    for sn in dataframe[sn_var].unique():
        for sid in dataframe.loc[dataframe[sn_var] == sn,sid_var].unique():
            for tp in dataframe.loc[(dataframe[sn_var] == sn) & (dataframe[sid_var] == sid), tp_var].unique():
                df_i =\
                    dataframe.loc[(dataframe[sn_var] == sn) & (dataframe[sid_var] == sid) & (dataframe[tp_var] == tp), :].reset_index(drop=True)
                if df_i.shape[0] > 0:
                    # assess timebase and alpha
                                # smoothing window adjusted to minutes
                    if (alpha == None) or (auto_calc_alpha == True):
                        timebase_sec =\
                            df_i['Timebase (s)'].mode().values[0]
                        print(smoothing_window_size_minutes, timebase_sec)
                        smoothing_window = int((60*smoothing_window_size_minutes)/timebase_sec)
                        alpha = 2/(smoothing_window+1)
                    else:
                        alpha=alpha
                    # print(alpha)
                    # initialize vars with first valid sample
                    new_sample=df_i.loc[0,variable] 
                    ema_previous= df_i.loc[0,variable] 
                    ema_ema_previous= df_i.loc[0,variable]
                    # create n_bypass based on approximate smothing window 
                    #  corresponding to alpha and timebase, with multiplier
                    #  where a multiplier of 1 gives you the standard approx smoothing window size
                    n_bypass_ema = (1/(alpha/2) - 1)
                    n_bypass_dema = (n_bypass_ema * 2) -1
                    # print(n_bypass_dema)
                    if n_bypass_dema % 1 > 0:
                        n_bypass_dema += n_bypass_dema
                    n_bypass_dema = int(n_bypass_dema)
                    for i in range(0, len(df_i[variable])):
                        # Refine the first ~ n samples to account for imbalanced weighting inherent in first few samples of DEMA
                        if i < n_bypass_dema:
                            ema, ema_ema, dema, ema_previous, ema_ema_previous =\
                                dema_smoother_FW_filter(
                                    i=i,
                                    alpha=alpha,
                                    new_sample=new_sample,
                                    ema_previous=ema_previous, 
                                    ema_ema_previous=ema_ema_previous,
                                    integer_math = integer_math
                                )
                            ## Overwrite DEMA with raw value, but keep calculating DEMA
                            dema = df_i.loc[i,variable]
                            ## drop data until n_bypass_dema samples have been taken
                            # new_sample = nan
                            # ema = nan
                            # ema_ema = nan
                            # dema = nan
                            # ema_previous = nan
                            # ema_ema_previous = nan
                            ## use raw data until n_bypass_dema samples have been taken
                            # new_sample = df_i.loc[i,variable]
                            # ema = df_i.loc[i,variable]
                            # ema_ema = df_i.loc[i,variable]
                            # dema = df_i.loc[i,variable]
                            # ema_previous = df_i.loc[i,variable]
                            # ema_ema_previous = df_i.loc[i,variable]
                        # elif i == n_bypass_dema:
                        #     mean_n_bypass_dema_agg = mean(n_bypass_dema_var_aggregator)
                        #     #
                        #     ema, ema_ema, dema, ema_previous, ema_ema_previous =\
                        #         dema_smoother_FW_filter(
                        #             i=i,
                        #             alpha=alpha,
                        #             new_sample=mean_n_bypass_dema_agg,
                        #             ema_previous=mean_n_bypass_dema_agg, 
                        #             ema_ema_previous=mean_n_bypass_dema_agg,
                        #             integer_math = integer_math
                        #         )
                        else:
                            new_sample = df_i.loc[i,variable]
                            ema, ema_ema, dema, ema_previous, ema_ema_previous =\
                                dema_smoother_FW_filter(
                                    i=i,
                                    alpha=alpha,
                                    new_sample=new_sample,
                                    ema_previous=ema_previous, 
                                    ema_ema_previous=ema_ema_previous,
                                    integer_math = integer_math
                                )
                    # for i in range (0,n_bypass_dema):
                    #     df_i.loc[i, output_var_name] = nan
                        # assign
                        # df_i.loc[i, variable + '_ema'] = ema_i
                        df_i.loc[i, output_var_name] = dema
                    # # # drop the first n datapoints where n = approximate smoothing window * 2
                    # n_bypass_dema = int(1/(alpha/2) - 1) * 2
                    # # print(n_bypass_dema)
                    # if n_bypass_dema % 1 > 0:
                    #     n_bypass_dema += n_bypass_dema
                    # for i in range (0,n_bypass_dema):
                    #     df_i.loc[i, output_var_name] = nan
                    if df_i.shape[0] >0:
                        df_i_list.append(df_i)
    # concat
    df_out = pd.concat(df_i_list, ignore_index=True)

    return df_out


def aae_calc(row, ln_babs_ratio):
    if row['b_abs_1'] == 0 or row['b_abs_2'] == 0:
        return np.nan
    elif row['b_abs_2'] / row['b_abs_1'] <= 0:
        return np.nan
    else:
        return -1 * ( np.log( row['b_abs_2'] / row['b_abs_1'] ) / ln_babs_ratio )


def sa_aethalometer_model(
        dataframe,
        wl_1='IR',
        wl_2='Blue',
        # wl_2='UV',
        var_type='BCc',
        output_smoothed_blue_bc = True,
        use_API_varnames = False,
        auto_estimate_timebase=True,
        timebase_sec=60,
        C=1.3,
        AAEff=1,
        AAEwb=2,
        smoothing_window_size_minutes=15,
        alpha=None,
        auto_calc_alpha = True,
        integer_math = True,
        map_column_names_to_standard_names=True,
        plot_sa_vars=False,
        limit_BB_to_between_0_to_100 = True,
        tag_data_snapped_to_0_and_100 = False,
        dev = 0,
        print_loading_bars = True
    ):
    """
    Post-process source apportionment caclulation.
    """

    if dev:
        # for dev - set variable defaults
        df_true = df.copy(deep=True) # save true origin df because we reassign "df" in the function
        dataframe = df_true.copy(deep=True)
        wl_1='IR'
        # wl_2='Blue'
        wl_2='UV'
        var_type='BC1'
        use_API_varnames = False
        timebase_sec=60
        C=1.3
        AAEff=1
        AAEwb=2
        smoothing_window_size_minutes=15
        alpha=None
        map_column_names_to_standard_names=True
        plot_sa_vars=False
        limit_BB_to_between_0_to_100 = True,
        tag_data_snapped_to_0_and_100 = False,
        integer_math=True
        # relative_tape_position_calc = apydp.relative_tape_position_calc

    df = dataframe.copy(deep=True)

    ### Assign variable names
    if not use_API_varnames:
        sn_var = 'Serial number'
        datetime_var = 'datetime_local'
        bcvar_wl1_assign = wl_1 + ' ' + var_type
        bcvar_wl1_bc1_assign = wl_1 + ' BC1'
        ir_bcc_var = 'IR BCc'
        blue_bcc_var = 'Blue BCc'
        ir_atn1_var = 'IR ATN1'
        tp_var = 'Tape position'
        test_var = 'test'
    else:
        sn_var = 'deviceId'
        datetime_var = 'unixtime'
        bcvar_wl1_assign =\
            'ma.wavelengths.{wl}.bc.{bc}'.format(wl=wl_1.lower(), bc=var_type.replace('BCc','bc').replace('BC1','b1').replace('BC2','b2'))
        bcvar_wl1_bc1_assign = bcvar_wl1_assign.replace('bc.bc','bc.b1').replace('bc.b2','bc.b1')
        ir_bcc_var = 'ma.wavelengths.ir.bc.bc'
        blue_bcc_var = 'ma.wavelengths.blue.bc.bc'
        ir_atn1_var = 'ma.wavelengths.ir.atn.a1'
        tp_var = 'Tape position'
        test_var = 'sessionId'

    ### Calculate all relative tape positions NOT by test
    if 'Relative tape position' not in df.columns:
        df_new =\
            relative_tape_position_calc(
                df,
                bytest = False,
                use_API_varnames = use_API_varnames
                )
        df_new['Relative tape position not by test'] = df_new['Relative tape position']
        #
        df =\
            df.merge(
                df_new[[sn_var,datetime_var,'Relative tape position not by test']]
            )
        del df_new
    else:
        df['Relative tape position not by test'] = df['Relative tape position']

    ### Drop IR BCc smoothed if already in dataframe
    if 'IR BCc smoothed' in df.columns:
        df = df.drop('IR BCc smoothed', axis=1)

    ### Perform SA calculation by device and discrete tape position; MA only
    snlist =\
        df.loc[(~ (df[sn_var].str.contains('AE51|AL30|AL60'))) & (~ (df[sn_var].str.contains('PAX'))), sn_var].unique()
    snlist.sort()
    df_out_list = []
    # if print_loading_bars:
    #     progress_bar_sn = tqdm(snlist, position=0, leave=True, desc="Calculating SA variables by device")
    # else:
    #     progress_bar_sn = snlist
    for sn in snlist:
        df_i = df.loc[df[sn_var] == sn, :].reset_index(drop=True)
        rtp_list_i = df_i['Relative tape position not by test'].unique()
        # print("\t\t%s" %rtp_list_i)
        rtp_list_i.sort()
        # print("\t\t%s" %rtp_list_i)
        if print_loading_bars:
            progress_bar_rtp = tqdm(rtp_list_i, position=0, leave=True, desc="\tProcessing %s by relative tape position" %sn)
        else:
            progress_bar_rtp = rtp_list_i
        for rtp in progress_bar_rtp:
            if print_loading_bars:
                print("\tProcessing %s - rTP %s" %(sn, rtp))
            df_j =\
                df_i.loc[
                    df_i['Relative tape position not by test'].astype(str) == (str(rtp)), 
                    :
                ].reset_index(drop=True)
            # constants
            wavelength_dict = {'Blue':470,'Green':528,'Red':625,'IR': 880, 'UV':375} 
            atn_cross_section_dict = {'Blue':19.070, 'Green':17.028,'Red':14.091,'IR': 10.120, 'UV':24.069}

            wl_1_nm = wavelength_dict[wl_1]
            wl_2_nm = wavelength_dict[wl_2]
            # print("WL2: %s (%s)" %(wl_2, wl_2_nm))

            babs_ratio = (wl_2_nm/wl_1_nm)
            ln_babs_ratio = float(str(np.log(babs_ratio))[0:14]) # to match fact that this is manually defined in firmware as -0.62718921276 for IR/Blue
            babs_ratio_neg_aFF = babs_ratio**(-AAEff)
            babs_ratio_neg_aWB = babs_ratio**(-AAEwb)

            sigma_1 = atn_cross_section_dict[wl_1]
            sigma_2 = atn_cross_section_dict[wl_2]
            # print("Sigma 2: %s" %sigma_2)

            # timebase
            if auto_estimate_timebase:
                # print('\n\n')
                # print(rtp_list_i)
                # print(df_j['Timebase (s)'])
                # print(df_j['Serial number'].head(5))
                # print(sn,rtp,df_j['Tape position'].unique())
                # print(df_j['Timebase (s)'].mode())
                timebase_sec =\
                    df_j['Timebase (s)'].mode().values[0]
                # print("\n\n")

            # smoothing window adjusted to minutes
            if alpha == None:
                smoothing_window = int((60*smoothing_window_size_minutes)/timebase_sec)
                alpha = 2/(smoothing_window+1)
            else:
                alpha=alpha

            ### If var_type == BCc, perform BC1 calculation for when IR ATN1 < 3 for BCc
            if var_type == 'BCc':
                bcvars_for_calc = [bcvar_wl1_assign, bcvar_wl1_bc1_assign]
            else:
                bcvars_for_calc = [bcvar_wl1_assign]
            #
            df_j_var_type_dict = {}
            # progress_bar_vartypes = tqdm(bcvars_for_calc, position=0, leave=True, desc="\tProcessing by BC var type")
            for var_type_i in bcvars_for_calc:
                df_j_i = df_j.copy(deep=True)
                ### DEMA smooth input BC var
                if use_API_varnames:
                    var_type_type =\
                        var_type_i[-2:].replace('bc','BCc').replace('b1','BC1').replace('b2','BC2')
                else:
                    var_type_type = var_type_i[-3:]
                bcvar_wl1 = var_type_i
                bcvar_wl2 = var_type_i.replace(wl_1.lower(),wl_2.lower()).replace(wl_1,wl_2)
                smoothed_bcvar_name_wl1 = bcvar_wl1 + ' smoothed'
                smoothed_bcvar_name_wl2 = bcvar_wl2 + ' smoothed'
                # print("WL2 and wl2 smoothed %s and %s" %(bcvar_wl2, smoothed_bcvar_name_wl2))
                #
                df_j_i=\
                    dema_smoother_FW_implementation(
                        dataframe = df_j_i,
                        variable = bcvar_wl1,
                        output_var_name = smoothed_bcvar_name_wl1,
                        alpha = alpha,
                        auto_calc_alpha = False,
                        smoothing_window_size_minutes = smoothing_window_size_minutes,
                        integer_math=integer_math
                    )
            
                df_j_i=\
                    dema_smoother_FW_implementation(
                        dataframe = df_j_i,
                        variable = bcvar_wl2,
                        output_var_name = smoothed_bcvar_name_wl2,
                        alpha = alpha,
                        auto_calc_alpha = False,
                        smoothing_window_size_minutes = smoothing_window_size_minutes,
                        integer_math=integer_math
                    )

                ### Calculate absorption coefficients
                df_j_i['b_abs_1'] = df_j_i[smoothed_bcvar_name_wl1] * (sigma_1 / C)
                df_j_i['b_abs_2'] = df_j_i[smoothed_bcvar_name_wl2] * (sigma_2 / C)

                ### Calculate babs_ff_1
                df_j_i['b_abs_ff_1_numer'] = df_j_i['b_abs_2'] - (df_j_i['b_abs_1'] * babs_ratio_neg_aWB)
                df_j_i['b_abs_ff_1_denom'] = babs_ratio_neg_aFF - babs_ratio_neg_aWB
                df_j_i['b_abs_ff_1'] = df_j_i['b_abs_ff_1_numer'] / df_j_i['b_abs_ff_1_denom']  
                # print("babs_ratio_neg_aFF: %s" %babs_ratio_neg_aFF)
                # print("babs_ratio_neg_aWB: %s" %babs_ratio_neg_aWB)

                ### Calculate babs_wb_1
                df_j_i['b_abs_wb_1'] = df_j_i['b_abs_1'] - df_j_i['b_abs_ff_1']

                ### Helin et al 2018
                df_j_i['bb'] = (df_j_i['b_abs_wb_1'] / df_j_i['b_abs_1'])
                # Limit BB to between 0 and 1
                if limit_BB_to_between_0_to_100:
                    # Tag datapoints affected
                    if tag_data_snapped_to_0_and_100:
                        df_j_i['bb_forced_to_between_0_and_100'] = 0
                        df_j_i.loc[
                                (df_j_i['bb'] < 0) | (df_j_i['bb'] > 100),
                            'bb_forced_to_between_0_and_100'] = 1
                    # Snap data to 0 and 1
                    df_j_i.loc[df_j_i['bb'] < 0, 'bb'] =\
                        0.0
                    df_j_i.loc[df_j_i['bb'] > 1, 'bb'] =\
                        1.0

                ### Calculate aae
                # print(sn, rtp, (df_j_i['b_abs_2'] / df_j_i['b_abs_1']).min(), (df_j_i['b_abs_2'] / df_j_i['b_abs_1']).max())
                # note -- "invalid value entered into log" will arise when df_j_i['b_abs_2'] / df_j_i['b_abs_1'] <= 0. This is outputs nan and is thereby ok.

                # calculate aae by applying the function aae_calc() to every row in df_j_i
                df_j_i['aae'] =\
                    df_j_i.apply(aae_calc,ln_babs_ratio=ln_babs_ratio, axis=1)

                ### Claculate BCc from FF and WB
                df_j_i['bc_wb'] = np.trunc((df_j_i['bb']) * df_j_i[smoothed_bcvar_name_wl1])
                df_j_i['bc_ff'] = np.trunc(( 1 - (df_j_i['bb']) ) * df_j_i[smoothed_bcvar_name_wl1])

                ### Convert BB%
                df_j_i['bb'] = np.trunc((df_j_i['bb'] * 100))

                ### Calculate delta C
                df_j_i['Delta-C'] =\
                    df_j_i[smoothed_bcvar_name_wl2] - df_j_i[smoothed_bcvar_name_wl1]

                ### DEMA smoothed IR BCc
                df_j_i =\
                    dema_smoother_FW_implementation(
                        dataframe=df_j_i,
                        variable = ir_bcc_var,
                        output_var_name = ir_bcc_var + " smoothed",
                        alpha = alpha,
                        integer_math=integer_math
                    )
                ### DEMA smoothed Blue BCc
                if output_smoothed_blue_bc:
                    df_j_i =\
                        dema_smoother_FW_implementation(
                            dataframe=df_j_i,
                            variable = blue_bcc_var,
                            output_var_name = blue_bcc_var + " smoothed",
                            alpha = alpha,
                            integer_math=integer_math
                        )
                    # dema_smoother(
                    #     df_j_i,
                    #     alpha = alpha,
                    #     variable = "IR BCc",
                    #     min_periods = smoothing_window_size_minutes
                    # )
                
                ### add dataframe to dict
                df_j_var_type_dict[var_type_type] = df_j_i
            
            ### assign df_j_out based on ATN threshold
            sa_vars = [
                    'aae','b_abs_1','b_abs_2',
                    'bc_wb','bc_ff','bb','Delta-C'
                ]
            if tag_data_snapped_to_0_and_100:
                sa_vars.append('bb_forced_to_between_0_and_100')
            if len(df_j_var_type_dict) > 1:
                df_j_bc1 = df_j_var_type_dict['BC1']
                df_j_bcc = df_j_var_type_dict['BCc']
                df_j_bc1[ir_atn1_var] = df_j_bc1[ir_atn1_var].astype(float)
                df_j_bcc[ir_atn1_var] = df_j_bcc[ir_atn1_var].astype(float)
                #
                df_j_bc1_j =\
                    df_j_bc1.loc[
                        df_j_bc1[ir_atn1_var] < 3,
                        list(set([sn_var,test_var,tp_var,datetime_var])) + sa_vars + [ir_bcc_var + ' smoothed'] + [blue_bcc_var + ' smoothed']
                    ].reset_index(drop=True)
                df_j_bcc_j =\
                    df_j_bcc.loc[
                        df_j_bcc[ir_atn1_var] >= 3,
                        list(set([sn_var,test_var,tp_var,datetime_var])) + sa_vars + [ir_bcc_var + ' smoothed'] + [blue_bcc_var + ' smoothed']
                    ].reset_index(drop=True)
                # print(df_j_bc1_j)
                # print(df_j_bcc_j)
                df_j_out = pd.concat([df_j_bc1_j,df_j_bcc_j], ignore_index=True)
            else:
                df_j_out =\
                    df_j_var_type_dict[var_type][
                        list(set([sn_var,test_var,tp_var,datetime_var])) + sa_vars + [ir_bcc_var + ' smoothed'] + [blue_bcc_var + ' smoothed']
                        ]
            # aggregate
            df_out_list.append(df_j_out)
    # recombine
    df_out = pd.concat(df_out_list, ignore_index=True)
    
    # # merge status data back in
    # if 'Status' in df.columns:
    #     df_out =\
    #         merge(
    #             left= df_out,
    #             right = df[[sn_var,datetime_var,'Status']],
    #             how='left'
    #         )

    ### Create metadata
    df_out['Cref'] = C
    df_out['AAE biomass'] = AAEwb
    df_out['AAE fossil fuel'] = AAEff

    if map_column_names_to_standard_names:
        # rename vars
        df_out.rename(
            axis=1,
            inplace=True,
            mapper={
                'aae' : 'AAE calculated',
                'bc_wb':'Biomass ' + var_type,
                'bc_ff':'Fossil fuel ' + var_type,
                'bb':'BB percent'
            }
        )

    # plot
    if plot_sa_vars:
        import plotly.express as px
        if use_API_varnames():
            print("\nPrinting SA Plots not Currently Supported when use_API_varnames = True\n")
        else:
            #
            df_out_sa_long =\
                df_out[[
                    sn_var,datetime_var,'test','Tape position',
                    'AAE calculated',
                    'BB percent',
                    'Biomass ' + var_type,
                    'Fossil fuel ' + var_type,
                    'Delta-C'
                ]].melt(id_vars = [sn_var,datetime_var,'test','Tape position'])
            #
            fig_sa = px.scatter(
                df_out_sa_long,
                x=datetime_var,
                y='value',
                color=sn_var,
                facet_row = 'variable',
                title='Source Apportionment data across all instruments',
                height=1200
            ).update_xaxes(matches=None).update_yaxes(matches=None)
            #
            fig_sa.show()

    return df_out


def capture_var_from_lastMetadata(
        lastMetadata_datapoint,
        vari='timebase',
        replace_None_with_0 = False
    ):
    
    if not pd.isna(lastMetadata_datapoint):
        lastMetadata_dict =\
            eval(str(lastMetadata_datapoint).replace('null','None').replace('true','True').replace('false','False'))
            # jsonloads(str(lastMetadata_datapoint))
        # print(lastMetadata_dict)
        if vari in lastMetadata_dict.keys():
            if (lastMetadata_dict[vari] == None) & (replace_None_with_0):
                lastMetadata_dict[vari] = 0
            return lastMetadata_dict[vari]
        else:
            if replace_None_with_0:
                return 0
            else:
                return None
    else:
        if replace_None_with_0:
            return 0
        else:
            return None


def WF_apiformat_variable_extraction(
        dataframe        
    ):
    
	# Make typical MAx formatted variables from the API vars

    df = dataframe.copy(deep=True)

    ### Create key new variables
    ## Add Session ID
    df['Session ID'] =\
        df['lastMetadata'].apply(
            capture_var_from_lastMetadata,
            vari='ma.sessionId'
        )
    ## Add timebase
    df['Timebase (s)'] =\
        df['lastMetadata'].apply(
            capture_var_from_lastMetadata,
            vari='ma.timebase'
        )
    ## Add flow setpoint
    df['Flow setpoint (mL/min)'] =\
        df['lastMetadata'].apply(
            capture_var_from_lastMetadata,
            vari='ma.flowSetpoint'
        )
    ## Add timezoneOffset
    df['Timezone offset (mins)'] =\
        df['lastMetadata'].apply(
            capture_var_from_lastMetadata,
            vari='timezoneOffset'
        )
    ## Add MA firmware version
    df['Firmware version'] =\
        df['lastMetadata'].apply(
            capture_var_from_lastMetadata,
            vari='ma.firmwareVersion'
        )
    ## Add WF firmware version
    df['WF firmware version'] =\
        df['lastMetadata'].apply(
            capture_var_from_lastMetadata,
            vari='firmwareVersion'
        )
    ## Add optical config
    df['Optical config'] =\
        df['lastMetadata'].apply(
            capture_var_from_lastMetadata,
            vari='ma.opticalConfig'
        )

    return df



def APIvarnames_to_originalvarnames_converter(
        dataframe,
        reorder=False
    ):
    
    orig_columns = list(dataframe.columns)
    
    new_MA_api_vars_without_a_match = [
            'id','device_type','deviceId:dataSeriesId','deviceId:datasetId',
            'deviceFamily','datetime','transfer','reportingPeriod','sourceMethod', 
            'update'
        ]

    conversion_dict={
        'ma.firmware':'Firmware version',
        'deviceType':'device_type', 
        'ma.deviceId': 'Serial number',
        'deviceId': 'Serial number',
        'lastMetadata.reportingPeriod':'Timebase (s)',
        'datetime_local':'datetime_local',
        'datetime_rounded_1min':'datetime_rounded_1min',
        'timezoneOffset':'Timezone offset (mins)',
        'gps.latitude': 'GPS lat (ddmm.mmmmm)',
        'gps.longitude': 'GPS long (dddmm.mmmmm)',
        'ma.gps.latitude': 'GPS lat (ddmm.mmmmm)',
        'ma.gps.longitude': 'GPS long (dddmm.mmmmm)',
        'ma.datumId':'Datum ID',
        'datumId':'Datum ID API',
        'ma.sessionId':'Session ID',
        #'sessionId':'Session ID',
        'ma.tapePosition':'Tape position',
        'ma.status':'Status',
        'ma.batteryRemaining':'Battery remaining (%)',
        'ma.opticalConfig': "Optical config",
        
        'ma.env.sample.rh':'Sample RH (%)',
        'ma.env.sample.temp':'Sample temp (C)',
        'ma.env.sample.dewpoint':'Sample dewpoint (C)',

        'ma.env.internal.temp':'Internal temp (C)',
        'ma.env.internal.pressure':'Internal pressure (Pa)',

        'ma.flow.setpoint': 'Flow setpoint (mL/min)',
        'ma.flow.total':'Flow total (mL/min)',
        'ma.flow.f1':'Flow1 (mL/min)',
        'ma.flow.f2':'Flow2 (mL/min)',

        'ma.wavelengths.ir.sensor.ref':'IR Ref',
        'ma.wavelengths.ir.sensor.s1':'IR Sen1',
        'ma.wavelengths.ir.sensor.s2':'IR Sen2',    
        'ma.wavelengths.ir.atn.a1':'IR ATN1',
        'ma.wavelengths.ir.atn.a2':'IR ATN2', 
        'ma.wavelengths.ir.atn.k':'IR K',
        'ma.wavelengths.ir.atn.ak':'IR K',
        'ma.wavelengths.ir.bc.b1':'IR BC1',
        'ma.wavelengths.ir.bc.b2':'IR BC2',
        'ma.wavelengths.ir.bc.c':'IR BCc',
        'ma.wavelengths.ir.bc.bc':'IR BCc',
        'ma.wavelengths.ir.bc.b1Smooth': 'IR BC1 smoothed  (ng/m^3)',
        'ma.wavelengths.ir.bc.bcSmooth': 'IR BCc smoothed  (ng/m^3)',

        'ma.wavelengths.red.sensor.ref':'Red Ref',
        'ma.wavelengths.red.sensor.s1':'Red Sen1',
        'ma.wavelengths.red.sensor.s2':'Red Sen2',    
        'ma.wavelengths.red.atn.a1':'Red ATN1',
        'ma.wavelengths.red.atn.a2':'Red ATN2', 
        'ma.wavelengths.red.atn.k':'Red K',
        'ma.wavelengths.red.atn.ak':'Red K',
        'ma.wavelengths.red.bc.b1':'Red BC1',
        'ma.wavelengths.red.bc.b2':'Red BC2',
        'ma.wavelengths.red.bc.c':'Red BCc',
        'ma.wavelengths.red.bc.bc':'Red BCc',
        'ma.wavelengths.red.bc.b1Smooth': 'Red BC1 smoothed  (ng/m^3)',
        'ma.wavelengths.red.bc.bcSmooth': 'Red BCc smoothed  (ng/m^3)',

        'ma.wavelengths.green.sensor.ref':'Green Ref',
        'ma.wavelengths.green.sensor.s1':'Green Sen1',
        'ma.wavelengths.green.sensor.s2':'Green Sen2',    
        'ma.wavelengths.green.atn.a1':'Green ATN1',
        'ma.wavelengths.green.atn.a2':'Green ATN2', 
        'ma.wavelengths.green.atn.k':'Green K',
        'ma.wavelengths.green.atn.ak':'Green K',
        'ma.wavelengths.green.bc.b1':'Green BC1',
        'ma.wavelengths.green.bc.b2':'Green BC2',
        'ma.wavelengths.green.bc.c':'Green BCc',
        'ma.wavelengths.green.bc.bc':'Green BCc',
        'ma.wavelengths.green.bc.b1Smooth': 'Green BC1 smoothed  (ng/m^3)',
        'ma.wavelengths.green.bc.bcSmooth': 'Green BCc smoothed  (ng/m^3)',

        'ma.wavelengths.blue.sensor.ref':'Blue Ref',
        'ma.wavelengths.blue.sensor.s1':'Blue Sen1',
        'ma.wavelengths.blue.sensor.s2':'Blue Sen2',    
        'ma.wavelengths.blue.atn.a1':'Blue ATN1',
        'ma.wavelengths.blue.atn.a2':'Blue ATN2', 
        'ma.wavelengths.blue.atn.k':'Blue K',
        'ma.wavelengths.blue.atn.ak':'Blue K',
        'ma.wavelengths.blue.bc.b1':'Blue BC1',
        'ma.wavelengths.blue.bc.b2':'Blue BC2',
        'ma.wavelengths.blue.bc.c':'Blue BCc',
        'ma.wavelengths.blue.bc.bc':'Blue BCc',
        'ma.wavelengths.blue.bc.b1Smooth': 'Blue BC1 smoothed  (ng/m^3)',
        'ma.wavelengths.blue.bc.bcSmooth': 'Blue BCc smoothed  (ng/m^3)',
                    
        'ma.wavelengths.uv.sensor.ref':'UV Ref',
        'ma.wavelengths.uv.sensor.s1':'UV Sen1',
        'ma.wavelengths.uv.sensor.s2':'UV Sen2',    
        'ma.wavelengths.uv.atn.a1':'UV ATN1',
        'ma.wavelengths.uv.atn.a2':'UV ATN2', 
        'ma.wavelengths.uv.atn.k':'UV K',
        'ma.wavelengths.uv.atn.ak':'UV K',
        'ma.wavelengths.uv.bc.b1':'UV BC1',
        'ma.wavelengths.uv.bc.b2':'UV BC2',
        'ma.wavelengths.uv.bc.c':'UV BCc',
        'ma.wavelengths.uv.bc.bc':'UV BCc',
        'ma.wavelengths.uv.bc.b1Smooth': 'UV BC1 smoothed  (ng/m^3)',
        'ma.wavelengths.uv.bc.bcSmooth': 'UV BCc smoothed  (ng/m^3)',
        
        'scd.relative_humidity':'Sample RH (%) - scd',
        'scd.temperature':'Sample temp (C) - scd',

        'ma.gps.accelX':'Accel X',
        'ma.gps.accelY':'Accel Y',
        'ma.gps.accelZ':'Accel Z',
        'gps.satCount': "GPS sat count",
        'gps.accelX':'Accel X',
        'gps.accelY':'Accel Y',
        'gps.accelZ':'Accel Z',

        'App Version':'App Version',
        'App version':'App Version',

        'ma.sourceApportionment.cref': 'Cref',
        'ma.sourceApportionment.aaeWoodBurning': 'AAE biomass',
        'ma.sourceApportionment.aaeFossilFuel' : 'AAE fossil fuel',
        'ma.sourceApportionment.bccWoodBurning' : 'Biomass BCc  (ng/m^3)',
        'ma.sourceApportionment.bccFossilFuel': 'Fossil fuel BCc  (ng/m^3)',
        'ma.sourceApportionment.aae' : 'AAE',
        'ma.sourceApportionment.bb': 'BB (%)',
        'ma.sourceApportionment.deltaC': 'Delta-C  (ng/m^3)',

        'ma.flow.pumpDrive': 'Pump drive',
        'ma.reportingTemp': "Reporting Temp (C)",
        'ma.reportingPressure': "Reporting Pressure (Pa)",
        'ma.wifiRssi':'WiFi RSSI',
        'ma.wifi_rssi':'WiFi RSSI',

        }

    # rename variables in dataframe to match MAx format
    for vari in conversion_dict.keys():
        if vari in dataframe.columns:
            dataframe.rename(
                columns={
                        vari: conversion_dict[vari]
                    }, inplace=True)

    if reorder:
            # reorder variables in dataframe to a cleaner format
            unchanged_vars = [x for x in orig_columns if x not in list(conversion_dict.keys())]
            vars_reordered = list(conversion_dict.values()) + unchanged_vars
            # print(unchanged_vars)
            # print(vars_reordered)
            dataframe = dataframe[vars_reordered]

    # # create new variables
    # dataframe['ratio_flow'] =\
    #     dataframe['Flow1 (mL/min)']/dataframe['Flow2 (mL/min)']
    # dataframe['Delineated status'] = dataframe['Status'].apply(bitwise_reducer_for_Status)
    # dataframe['Readable status'] = dataframe['Delineated status'].apply(status_number_translator, key_notes_only = trunacte_readable_status)

    return dataframe



def bitwise_reducer_and_translator_for_Status(
        n,
        drop_non_customer_facing_values = True,
        fw_level_format = True,
        output_as_string = False
    ):

    ### MOST UP TO DATE -- USE THIS ONE WHERE POSSIBLE, May 2023 ###

    ### define all status codes
    ## Internal format
    status_code_dict = {
        2** 0: 'DATUM_STATUS_ALL_CLEAR',  # not in mAM/ not shown to customer
        2** 1: 'DATUM_STATUS_INSTRUMENT_START_UP',
        2** 2: 'DATUM_STATUS_TAPE_ADVANCE',
        2** 3: 'DATUM_STATUS_DATA_ZERO', # not in mAM/ not shown to customer;  'Sen1 Sen2 and Ref are all 0 for a given wavelength'
        2** 4: 'DATUM_STATUS_DATA_SATURATED', # Sen or Ref saturation detected
        2** 5: 'DATUM_STATUS_CYCLE_TOO_LONG', # this may or may not manifest as too many or too few values in the numerator without adjustment of the denominator resulting in, for exaxmple, very large or very small Sen values; Data sampling timing cycle out of sync but data may or may not be affected | look for unreasonably large or small values among any and all variables
        2** 6: 'DATUM_STATUS_SPOT2_ACTIVE', 
        2** 7: 'DATUM_STATUS_FLOW_UNSTABLE',
        2** 8: 'DATUM_STATUS_FLOW_RANGE', # Flow range is just the pump drive level. It knows nothing of flow meters. If the pump is really working hard, we get that error
        2** 9: 'DATUM_STATUS_TIME_SOURCE_MANUAL', # time set manually
        2** 10: 'DATUM_STATUS_NO_TA_AT_START_UP',
        
        2** 11: 'DATUM_STATUS_SYSTEM_BUSY',

        2** 12: 'DATUM_STATUS_SA_NO_EN',

        2** 13: 'DATUM_STATUS_TAPE_JAM',
        2** 14: 'DATUM_STATUS_TAPE_AT_END',
        2** 15: 'DATUM_STATUS_TAPE_NOT_READY',
        2** 16: 'DATUM_STATUS_TRANSPORT_NOT_READY',
        
        2** 17: 'DATUM_STATUS_EXT_5V',
        2** 18: 'DATUM_STATUS_RTC_DATE_TIME_BAD', # When the clock has to be initialized, or is initialized it is set to the year 2000 and rtc time is flagged as bad; It stays bad until the clock is set again by GPS or manually; 
        

        2** 19: 'DATUM_STATUS_TAPE_ERROR',    

        2** 20: '', # not in mAM / not shown to customer    
        2** 21: '', # not in mAM / not shown to customer    
        2** 22: '', # not in mAM / not shown to customer   
        2** 23: '', # not in mAM / not shown to customer

        2** 24: 'DATUM_STATUS_WIFI_MIN_TIMEBASE_60',
        2** 25: 'DATUM_STATUS_WIFI_COMPLETION_0', # not in mAM / not shown to customer
        2** 26: 'DATUM_STATUS_WIFI_COMPLETION_1', # not in mAM / not shown to customer
        2** 27: 'DATUM_STATUS_WIFI_COMPLETION_2', # not in mAM / not shown to customer
        2** 28: 'DATUM_STATUS_WIFI_COMPLETION_3', # not in mAM / not shown to customer
        2** 29: 'DATUM_STATUS_WIFI_COMPLETION_4', # not in mAM / not shown to customer
        2** 30: 'DATUM_STATUS_WIFI_COMPLETION_5', # not in mAM / not shown to customer
        2** 31: 'DATUM_STATUS_WIFI_COMPLETION_6',# not in mAM / not shown to customer
        2** 32: 'DATUM_STATUS_WIFI_LINE_FULL', # not in mAM / not shown to customer
        2** 33: '', # not in mAM / not shown to customer

        2** 34: '', # not in mAM / not shown to customer
        2** 35: 'DATUM_STATUS_TEST_DATUM_HI', # not in mAM / not shown to customer
        2** 38: 'DATUM_STATUS_REMOTE_PWD',
    }
    # Drop internal-only diagnostic variables
    if drop_non_customer_facing_values:
        diagnostic_only_keys_exponentvals = [
            0,
            3,
            6,
            20, 21, 22, 23,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        ]
        for exp_i in diagnostic_only_keys_exponentvals:
            status_code_dict[2**exp_i] = ""
    
    ## Firmware format
    status_code_human_readable_dict_mAM_format = {
        2** 0: '',
        2** 1: 'Start up',
        2** 2: 'Tape advance',
        2** 3: '',
        2** 4: 'Optical saturation',
        2** 5: 'Sample timing error', # this may or may not manifest as too many or too few values in the numerator without adjustment of the denominator resulting in, for exaxmple, very large or very small Sen values
        ###

        2** 6: 'DualSpot on',
        2** 7: 'Flow unstable',
        2** 8: 'Pump out of range',
        2** 9: 'Time source manual',
        2** 10: 'User skipped tape advance',
        
        2** 11: 'System busy',

        2** 12: 'S.A. disabled',
        
        2** 13: 'Tape jam',
        2** 14: 'Tape at end',
        2** 15: 'Tape not ready',
        2** 16: 'Tape transport not ready',

        2** 17: 'Ext. power',
        2** 18: 'Invalid date/time', # When the clock has to be initialized, or is initialized it is set to the year 2000 and rtc time is flagged as bad; It stays bad until the clock is set again by GPS or manually
        
        2** 19: 'Tape error',

        2** 20: '',
        2** 21: '',
        2** 22: '',
        2** 23: '',
            
        ### Unclear which if any WiFi codes will be exposed to customers
        2** 24: 'WiFi min timebase 60',
        # 2** 25: 'WiFi check 0 completed',
        # 2** 26: 'WiFi check 1 completed',
        # 2** 27: 'WiFi check 2 completed',
        # 2** 28: 'WiFi check 3 completed',
        # 2** 29: 'WiFi check 4 completed',
        # 2** 30: 'WiFi check 5 completed',
        # 2** 31: 'WiFi check 6 completed',
        2** 32: 'WiFi line busy',

        2** 33: '',
        # 2** 34: '',
        
        # 2** 35: 'DATUM_STATUS_TEST_DATUM_HI' #Not shown in mAM
        2**38: 'Remote power down'

    }
    
    if fw_level_format:
        dict_to_use = status_code_dict
    else:
        dict_to_use = status_code_human_readable_dict_mAM_format
    # print(n)
    if (n == np.nan) or pd.isnan(n):
        status_vals_desc = []
    else:
        n = int(n)
        ### Identify values that contribute to the status
        status_vals_with0s = [n & x for x in dict_to_use.keys()]
        status_vals_list = list(set([x for x in status_vals_with0s if x != 0]))
        status_vals_desc_list = [dict_to_use[x] for x in status_vals_list]
        if status_vals_desc_list == []:
            status_vals_desc = ''
        else:
            status_vals_desc = sub("'|\[|\]",'',str(status_vals_desc_list))
    if output_as_string:
        status_vals_desc = str(status_vals_desc)

    return status_vals_desc



def assign_tape_position_from_readable_status(
        dataframe,
        overwrite_existing_tape_position = False,
        datetime_var = 'datetime_utc'
        ):

    df = dataframe[['Serial number',datetime_var,'Readable status']]
    # flag tape advances
    df['Tape advance process'] = 0
    df.loc[df['Readable status'].str.contains('Tape advance'),'Tape advance process'] = 1
    df['Tape advance process differential'] =\
        df['Tape advance process'].diff().fillna(0)
    df['Tape advance flag'] = 0
    df.loc[
            (df['Tape advance process'] == 1) &
            (df['Tape advance process differential'] == 1)
            ,
            'Tape advance flag'
        ] = 1
    df_i_list = []
    for sn in df['Serial number'].unique():
        df_i = df[df['Serial number'] == sn].reset_index(drop=True)
        # check for presence of tape position variable
        if ('Tape position' not in df.columns) | overwrite_existing_tape_position:
            df_i.sort_values(by=datetime_var,ascending=True,inplace=True)
            df_i['Tape position'] = df_i['Tape advance flag'].cumsum() + 1
        df_i_list.append(df_i)
    df_concat = pd.concat(df_i_list, ignore_index=True)
    # merge back
    df =\
        dataframe.merge(
                df_concat[['Serial number',datetime_var,'Tape position']],
                how='left',
                on=[datetime_var,'Serial number']
            )

    return df



def read_API_formatted_data_from_disk(
        filepath="",
        data_format="CSV",
        convert_colnames_to_match_MAx = False,
        create_new_MAx_vars = False,
        include_fw_level_status_values=True,
        verbose=False,
        num_mins_datetime_round=1,
        timezone_string="",
        assign_tape_position_from_Readable_status_MA=False,
        assign_tape_position_from_Readable_status_WF=True,
        drop_missingDatumIddata=True
    ):

    """
    THIS IS NOT YET ADAPTED FOR SCOUT DEVICES
    """

    ### Read data
    if data_format == "CSV":
        df = pd.read_csv(filepath, low_memory=False)
    
    ### Format datetime
    # UTC
    df['datetime_utc'] =\
        pd.to_datetime(df['unixtime'], unit='ms').dt.tz_localize('UTC')
    # Local time
    if len(timezone_string) > 0:
        try:
            df['datetime_local'] =\
                df['datetime_utc'].dt.tz_convert(timezone_string).dt.tz_localize(None)
        except Exception as e:
            print("Exception %s -- could not properly create datetime_local from timezone_string" %e)
    else:
        df['datetime_local'] = df['datetime_utc'].copy(deep=True)
        print("\nUNABLE TO PRODUCE datetime_local VARIABLE. USING UTC AS LOCAL TIMEZONE\n\tPlease define timezone_string argument.\n")
    # Create rounded datetime local
    df['datetime_rounded_'+str(num_mins_datetime_round) + 'min'] =\
        df.datetime_utc.dt.round(freq = str(num_mins_datetime_round) + 'min')
    
    ### Extract WF variables from lastStatus
    device_type = df['deviceType'].mode()[0]
    if ("wf" in device_type) or ("WF" in device_type):
        df = WF_apiformat_variable_extraction(dataframe = df)

    ### Convert column names to match MAx
    if convert_colnames_to_match_MAx:
        df =\
            APIvarnames_to_originalvarnames_converter(
                dataframe=df,
                reorder=False
            )
        flow1var = "Flow1 (mL/min)"
        flow2var= "Flow2 (mL/min)"
    else:
        flow1var = "ma.flow.f1"
        flow2var = "ma.flow.f2"
        if 'ma.status' in df.columns:
            df['Status'] = df['ma.status']

    ## Human-readable status
    if include_fw_level_status_values:
        df['Readable status - fw format'] =\
            df['Status'].apply(
                bitwise_reducer_and_translator_for_Status,
                    fw_level_format=True,
                    output_as_string = True
                    )
    df['Readable status'] =\
         df['Status'].apply(
            bitwise_reducer_and_translator_for_Status,
                fw_level_format=False,
                output_as_string = True
                )

    ## Assign tape position from Readable status columns
    if (
            (("wf" in device_type) or ("WF" in device_type)) and 
            (assign_tape_position_from_Readable_status_WF)
        ) or (
            ("MA" in device_type) and
            (assign_tape_position_from_Readable_status_MA)
        ):
        if 'Tape position' in df.columns:
            df.drop(columns=['Tape position'], inplace=True)
        df =\
            assign_tape_position_from_readable_status(
                    df,
                    overwrite_existing_tape_position = False,
                    datetime_var = 'datetime_utc'
                )

    ### Create new MAx variables as necessary
    if create_new_MAx_vars:
        # ratio_flow
        df['ratio_flow'] =df[flow1var] / df[flow2var]

    ### Drop where missingDatumId == True
    if drop_missingDatumIddata and ('missingDatumId' in df.columns):
        df = df[df['missingDatumId'] != True].reset_index(drop=True)

    ### Return df
    return df


def grab_testid_from_filename(
        filename, split_by = '_', object_num_to_use=-1, num_digits_to_use=8,
        file_extension=r".csv|CSV|.dat|.DAT"
    ):
    """
    Using internal AethLabs file naming convention, this creates a test id from the test date (pulled from the filename) as YYDDMM
    """
    
    filename_split = filename.split(split_by)
    testid = filename_split[object_num_to_use]
    if 'AE51' in filename:
        dev_type = 'AE51'
    if (('MA2' in filename) or ('MA3' in filename)) and ("BC1-MA350" not in filename):
        dev_type='MAx'
    if "BC1-MA350" in filename:
        dev_type='ClarityBC1'
    if 'PAX' in filename:
        dev_type='PAX'
    if 'AE33' in filename:
        dev_type = 'AE33'
    if dev_type == 'MAx':
        testid = sub(file_extension,"",testid)[0:num_digits_to_use]
    if dev_type == "ClarityBC1":
        testid = sub(file_extension,"",testid)[0:num_digits_to_use]
    if 'AE51' in dev_type:
        testid = sub(file_extension,"",testid)[2:num_digits_to_use+2]

    return testid


def readMAx(
        filepath,
        clean=True,
        sep=',',
        verbose=True,
        assign_testid_from_startdate=True,
        assign_testid_from_filename=False,
        split_by='_',
        object_num_to_use=-1,
        num_digits_to_use=6,
        num_mins_datetime_round=1,
        process_api_formatted_files=False
    ):
    """
    Reads a single MAx file direct from a filepath.

    Inputs:pilot
        - Path to unaltered MAx file (currently only .csv extensions)

    Outputs:
        - a Pandas dataframe with properly formatted datetime

    Noteworthy features:
        - User specifies file delimter; comma  is default
        - Creates local and UTC datetime columns in Python datetime format for
            convenience -- this method assumes there is a single offset for the
            entire dataset
        - assumes there is only a single time offset for the entire data file
        - The "clean" option truncates the full set of variabls to include
          only key variables of interest
          (dropping 'Date / time local',
          Date local (yyyy/MM/dd)','Time local (hh:mm:ss)').
    """

    # Determine file delimeter -- default as ","
    sep= sep
    with open(filepath) as ff:
        reader = csv_reader(ff)
        line1 = next(reader)    
    if ";" in str(line1):
        sep=";"
    ff.close()

    ### Convert column names to match MAx
    if process_api_formatted_files:
        # check if file is API format
        new_filepaths_ma = []
        f_head1 = pd.read_csv(filepath, nrows=1)
        f_cols = f_head1.columns
        if 'ma.wavelengths.ir.atn.a1' not in f_cols:
            file_in_API_format = False
            print('File in API format')
        else:
            file_in_API_format = True
        if file_in_API_format:
            df = read_API_formatted_data_from_disk(
                filepath=filepath,
                data_format="CSV",
                convert_colnames_to_match_MAx = True,
                create_new_MAx_vars = False,
                include_fw_level_status_values=True,
                verbose=verbose,
                num_mins_datetime_round=1,
                timezone_string="",
                assign_tape_position_from_Readable_status_MA=False,
                assign_tape_position_from_Readable_status_WF=True,
                drop_missingDatumIddata=True
            )
            # re-create datetime vars to be sure        
            df['datetime_utc'] = pd.to_datetime(df['unixtime'], unit='ms')
            try:
                df['datetime_local'] = df['datetime_utc'] + pd.to_timedelta(df['timezoneOffset'], unit='m',errors='coerce')
            except Exception as e:
                df['timezoneOffset'] = 0
                try:
                    df['datetime_local'] = df['datetime_utc'] + pd.to_timedelta(df['timezoneOffset'], unit='m',errors='coerce')
                except Exception as e:
                    df['datetime_local']= df['datetime_utc']
                    print(e)

            if verbose:
                print(df.head())
        else:
                # otherwise, read normally
            df = pd.read_csv(filepath, low_memory=False, sep=sep)
            # datetime
            df['datetime_local'] = pd.to_datetime(df['Date / time local'])
            # calcualte utc
            df['datetime_utc'] = df['datetime_local'] + \
                pd.to_timedelta(df['Timezone offset (mins)'], unit='m',errors='coerce')
    else:
        # otherwise, read normally
        df = pd.read_csv(filepath, low_memory=False, sep=sep)
        # datetime
        df['datetime_local'] = pd.to_datetime(df['Date / time local'])
        # calcualte utc
        df['datetime_utc'] = df['datetime_local'] + \
            pd.to_timedelta(df['Timezone offset (mins)'], unit='m',errors='coerce')
    ### Create new columns
    df['datetime_rounded_'+str(num_mins_datetime_round) + 'min'] = df.datetime_local.dt.round(freq = str(num_mins_datetime_round) + 'min')
    df['ratio_flow'] = df['Flow1 (mL/min)'] / df['Flow2 (mL/min)']
    df['device_type'] = df['Serial number'].str.split('-')[0][0]
    # single or dual spot?
    df['spot_mode'] = 'PROCESSING ERROR'
    try:
        df.loc[df['Optical config'].str.contains("SINGLE"), 'spot_mode'] = 'single'
        df.loc[df['Optical config'].str.contains("DUAL"), 'spot_mode'] = 'dual'
    except Exception as e:
        print("PROCCESSING ERROR WITH SPOT_MODE FROM OPTICAL CONFIG", e, "assuming NaN = DUAL")
        df['Optical config'] = df['Optical config'].fillna('DUAL')


    # Where Spot Mode is single spot, ensure Spot 2 related columns are empty and not 'na' or ' " " etc
    spot_2_specific_cols = ['IR ATN2','Red ATN2','Green ATN2','Blue ATN2','UV ATN2',
                            'IR BC2','Red BC2','Green BC2','Blue BC2','UV BC2',
                            'IR BCc','Red BCc','Green BCc','Blue BCc','UV BCc',
                            'IR K','Red K','Green K','Blue K','UV K',
                            'Flow2 (mL/min)']
    df.loc[df['Optical config'].str.contains("SINGLE"), spot_2_specific_cols] = np.nan
    #calculate deltaC -- allen et al 2004, Evaluation of a New Approach for Real Time Assessment of Wood Smoke PM
    df['deltaC1'] = df['UV BC1'] - df['IR BC1']
    df['deltaC2'] = df['UV BC2'] - df['IR BC2']
    df['deltaCc'] = df['UV BCc'] - df['IR BCc']
    # Calculate sen/ref vals
    df['IR Sen1/Ref'] = df['IR Sen1'] / df['IR Ref']
    df['IR Sen2/Ref'] = df['IR Sen2'] / df['IR Ref']
    df['Red Sen1/Ref'] = df['Red Sen1'] / df['Red Ref']
    df['Red Sen2/Ref'] = df['Red Sen2'] / df['Red Ref']
    df['Green Sen1/Ref'] = df['Green Sen1'] / df['Green Ref']
    df['Green Sen2/Ref'] = df['Green Sen2'] / df['Green Ref']
    df['Blue Sen1/Ref'] = df['Blue Sen1'] / df['Blue Ref']
    df['Blue Sen2/Ref'] = df['Blue Sen2'] / df['Blue Ref']
    df['UV Sen1/Ref'] = df['UV Sen1'] / df['UV Ref']
    df['UV Sen2/Ref'] = df['UV Sen2'] / df['UV Ref']

    if clean:
        drop_list = ['Date / time local',
                    'Date local (yyyy/MM/dd)', 'Time local (hh:mm:ss)']
        for var in drop_list:
            if var in df.columns:
                df.drop(var, axis=1, inplace=True)
    if assign_testid_from_filename:
        t = grab_testid_from_filename(filename=filepath, split_by = split_by, object_num_to_use=object_num_to_use,num_digits_to_use=num_digits_to_use)
        df['test'] = t
    if assign_testid_from_startdate:
        df['test']= str(df.datetime_local.dt.date[0])
    if verbose:
        print('MA file succsessfully loaded.')
    return df

def sn_session_id_maker(
        df,
        snvar='Serial number',
        sidvar='Session ID'
    ):

    # sort values to reduce fragmentation
    df.sort_values(by=[snvar, sidvar], inplace=True)

    # loop through snvar and sidvar to create new sn_session_id var
    sn_session_id_list =[]
    df_new_list =[]
    sn_progressbar = tqdm(df[snvar].unique(), position=0, leave=True, desc="\tCreating Session IDs")
    for sn in sn_progressbar:
        for sid in df.loc[df[snvar] == sn, sidvar].unique():
            df_new =\
                df.loc[
                    (
                        (df[snvar] == sn) &
                        (df[sidvar] == sid)
                    ), :].reset_index(drop=True)
            
            sn_session_id = str(sn)+'_'+str(sid)
            sn_session_id_list.append(sn_session_id)
            df_new['sn_session_id'] = sn_session_id
            #
            df_new_list.append(df_new)
    # recombine
    print("\t\tRecombining data files as part of Session ID creation (this can take several seconds or longer) ...")
    df_new = pd.concat(df_new_list).reset_index(drop=True)
            
    return df_new, sn_session_id_list

def sn_test_id_maker(
        df,
        sn_var='Serial number'
    ):
    sn_test_id_list =[]
    for sn in df[sn_var].unique():
        for test in df.loc[df[sn_var] == sn, 'test'].unique():
            sn_test_id = str(sn)+'_'+str(test)
            sn_test_id_list.append(sn_test_id)
            df.loc[(df[sn_var]== sn )& (df['test']==test),'sn_test_id'] = sn_test_id
            #
            df_new = df.copy()

    return df_new, sn_test_id_list

# Shortened and simplified version for MAx data only!
def readall_BCdata_from_dir(
        directory_path = None,
        sep = ",",
        mult_folders_in_dir = True,
        verbose = False,
        summary = False,
        AE51_devices_only = False,
        file_number_printout = True,
        output_pax_averaged_by_minute = True,
        PAX_correction = False,
        inter_device_corr = True,
        assign_testid_from_startdate = True,assign_testid_from_filename = False,
        split_by = '_',
        object_num_to_use = -1,
        num_digits_to_use = 6,
        num_mins_datetime_round = 1,
        internal_QAQC_summary_format = True,
        group_Session_Ids_together = False, 
        datetime_fixme_dict = {
                'Serial numbers':[],
                'timeoffsets_minutes':[]
            },
        assign_unit_category_from_dirname = False,
        test_batch_indication = False,
        oring_from_filename = False,
        allow_ALx_files = True,
        files_to_exclude = [],
        output_first_datapoint_of_each_file_separately = False,
        create_session_ids = True,
        process_api_formatted_files=True,
        sigmaatn_al30 = 6.776, # Circa April 2025 Kyan - similar to 6.6 from bond et al. 
        sigmaatn_al60 = 12.5/1.8,
        sigmaatn_al80 = 12.5/1,
    ):
	"""
	Applies readMAx() and/or readAE() and/or readPax() and/or read_ALx_beta to a collection of files
	in a directory either with or without subdirectories

	Inputs:
		- A long Pandas dataframe (data veritcally stacked) with timeseries
		data from more than one AE51, MAx, and/or PAX device types

	Outputs:
		- Returns either:
			- 1 dataframe: a long dataframe of all device data in that
				directory, outputted in the MAx format regardless of
				device type(s)
			- 2 dataframes: (if Summary is set to True) BOTH a long dataframe
				of all device data in that directory in the MAx format AND a
				dataframe with sumary information on all devices (implements
				multidevice_BCdata_sumary() and outputs summary info on one
				device per row of the summary dataframe)

	Noteworthy features:
		- mult_folders_in_dir (Default is True) is set to true if
			subdirectories with files of interest exist within the primary
			directory
		- sep is the delimiter used by the MAx file. By default, this should
			be a comma but in some countries becomes a semicolon
		- the "plots" option (Default=True)  outputs a timeseries plot of BC
			readings with a frame for each wavelength; it is saved to the
			designated filepath.
		- AE51_devices_only option can be True only when AE51 devices are being
			analyzed. This parameter, when True, alters the function to read data
			outputted by readAE() under the readAE() setting of ma_format==False.
			The dataframe and summary dataframes will have different variable
			names and a different format than those outputted when MAx files are
			involved or the AE51_devices_only option is not invoked.
		- Optionaly prints out the number of files by type to be read in via
			file_number_printout
		- optionally prints PAX data as 1-minute averages (simple arithmetic
			averages of 1hz data)
	"""

	categories = ['Production - Round 1', 'Production - Round 2 Plus',
					'Service - Round 1', 'Service - Round 2 Plus',
					'Other', 'Testers'
					]

	print("\n\t... Collecting file paths ...")

	if AE51_devices_only is False:
		if mult_folders_in_dir:
			filepaths_ma =\
				globglob(
					directory_path +\
						ossep +\
						'**' +\
						ossep +\
						'*MA[0-9]*.csv', 
					recursive=True
				)
			# Remove Clarity BC1 files that may have sneaked their way in
			filepaths_ma = [x for x in filepaths_ma if 'BC1-MA350' not in x]
			if verbose:
				print("MAx filepaths collected:") 
				for f in filepaths_ma:
					print("\t%s" % f)
			filepaths_ae =\
				globglob(
					directory_path +\
						ossep +\
						'**' +\
						ossep +\
						'*AE51*.dat',
					recursive=True
					)
			filepaths_ae_csv =\
				globglob(
					directory_path +\
						ossep +\
						'**' +\
						ossep +\
						'*AE51*.csv',
					recursive=True
					)
			filepaths_pax =\
				globglob(
					directory_path +\
						ossep +\
						'**' +\
						ossep +\
						'PAX-011_m*.csv',
					recursive=True
				)
			filepaths_ClarityBC1 =\
				globglob(
					directory_path +\
						ossep +\
						'**' +\
						ossep +\
						'BC1-MA350*.csv',
					recursive=True
				)
			filepaths_ALx =\
				globglob(
					directory_path +\
						ossep +\
						'**' +\
						ossep +\
						'AL[0-9]0*.json',
					recursive=True
				)
			# drop any ALx NEWFORMAT files from the list
			if len(filepaths_ALx) > 0:
				filepaths_ALx =\
					[f for f in filepaths_ALx if 'NEWFORMAT' not in f]
			# drop any MA files that are in API format from the list
			if (len(filepaths_ma) > 0) & (process_api_formatted_files == False):
				new_filepaths_ma = []
				for f in filepaths_ma:
					f_head1 = pd.read_csv(f, nrows=1)
					f_cols = f_head1.columns
					if 'ma.wavelengths.ir.atn.a1' not in f_cols:
						new_filepaths_ma.append(f)
				filepaths_ma = new_filepaths_ma
		else:
			filepaths_ae =\
				globglob(
					directory_path +\
						ossep +\
						'**' +\
						ossep +\
						'*AE51*.dat',
					recursive=True
					)
			filepaths_ae_csv =\
				globglob(
					directory_path +\
						ossep +\
						'**' +\
						ossep +\
						'*AE51*.csv',
					recursive=True
					)

		# check files against list of files to exclude
		if len(files_to_exclude) > 0:
			print("\tExcluding files:\n")
			for f in files_to_exclude:
				print("\t\t%s" %f)
			filepaths_ma = [f for f in filepaths_ma if f not in files_to_exclude]
			filepaths_ae = [f for f in filepaths_ae if f not in files_to_exclude]
			filepaths_ae_csv = [f for f in filepaths_ae_csv if f not in files_to_exclude]
			filepaths_pax = [f for f in filepaths_pax if f not in files_to_exclude]
			filepaths_ClarityBC1 = [f for f in filepaths_ClarityBC1 if f not in files_to_exclude]
			filepaths_ALx = [f for f in filepaths_ALx if f not in files_to_exclude]

		# print error if AE51 .csv files are observed
		if len(filepaths_ae_csv) >0:
			print('\n\n!! One or more AE51 files has been downloaded in .csv format, which cannot be read by this tool !!\n\n%s' %filepaths_ae_csv)

		df_list = []
		x = 1
		xx = 1
		y = 1
		yy = 1
		z = 1
		first_data_point_list = []
		if file_number_printout and (not AE51_devices_only):
			string_to_print_fileload = '\n\tTotal files to be read: %s MA files, %s AE51 files, %s PAX files, %s Clarity BC1 files' % (len(filepaths_ma), len(filepaths_ae), len(filepaths_pax), len(filepaths_ClarityBC1))   
			if allow_ALx_files:
				string_to_print_fileload = string_to_print_fileload + ', and %s ALx files' % len(filepaths_ALx)
			print(string_to_print_fileload)

		if len(filepaths_ma) > 0:
			progress_bar_ma = tqdm(filepaths_ma, position=0, leave=True, desc="\tReading MAx files")
			for f in progress_bar_ma:    
				try:
					if verbose:
						print('Reading ',x, 'out of %d MA device files' % len(filepaths_ma))
						print('Reading:',f)
					# check to see if file is in API format
					f_head1 = pd.read_csv(f, nrows=1)
					f_cols = f_head1.columns
					if 'ma.wavelengths.ir.atn.a1' in f_cols:
						dft0 = readMAx(
							f, sep=sep, verbose=verbose,
							assign_testid_from_startdate=assign_testid_from_startdate,
							assign_testid_from_filename=assign_testid_from_filename,
							split_by=split_by,object_num_to_use=object_num_to_use,
							num_digits_to_use=num_digits_to_use, num_mins_datetime_round=num_mins_datetime_round,
							process_api_formatted_files=process_api_formatted_files
						)
					else:
						dft0 = readMAx(
							f, sep=sep, verbose=verbose,assign_testid_from_startdate=assign_testid_from_startdate,
							assign_testid_from_filename=assign_testid_from_filename, split_by=split_by,
							object_num_to_use=object_num_to_use,num_digits_to_use=num_digits_to_use,
							num_mins_datetime_round=num_mins_datetime_round,process_api_formatted_files=False
						)
					# assign category type (internal QAQC related)
					if assign_unit_category_from_dirname:
						for category in categories:
							if category in f:
								dft0['category'] = category
					# assign test batch
					if test_batch_indication:
						filename_features = f.split(ossep)
						batch_id_full = [x for x in filename_features if "Batch" in x][0]
						batch_id = sub('Test Batch ','',batch_id_full)
						dft0['Test Batch'] = batch_id
					if oring_from_filename == True:
						# look for oring designation
						if "no oring" in f:
							dft0['oring'] = 0
						else:
							dft0['oring'] = 1
					# Sort and capture first datapoint
					# print([x for x in dft0.columns if 'ate' in x])
					if 'datetime_local' in dft0.columns:
						dft0 =\
							dft0.sort_values(by="datetime_local").reset_index(drop=True)
					first_data_point_list.append(dft0.iloc[0])
					df_list.append(dft0)
					x += 1
				except Exception as e:
					x += 1
					print('\nException, could not load %s\n\t%s\n' %(f,e))
					pass

		print("\tConcatenating data into a single dataframe ...")
		df_compilation = pd.DataFrame([])
		try:
			df_compilation = pd.concat(df_list, sort=True)
		except Exception as e:
			print("Concatenation error:", e)
			# for i, df in enumerate(df_list):
			#     if df.columns.duplicated().any():
			#         print(f"Duplicate columns found in df_list[{i}]:", df.columns[df.columns.duplicated()])
			#         #return df_list
			try:
				for i, df_api in enumerate(df_list):
					if df_api.columns.duplicated().any():
						print(f"Fixing duplicate columns in df_list[{i}]...")
						df_list[i] = df_api.loc[:, ~df_api.columns.duplicated()]
					df_compilation = pd.concat(df_list, sort=True)
			except Exception as e:
				print("Concatenation error:", e)
				return df_list


		print('\t\tConcatenation complete.')
		# Convert first datapoint per file into dataframe
		df_first_datapoint_per_file =\
			pd.DataFrame(first_data_point_list).reset_index(drop=True).rename(columns={'index':'datetime_local','0':'test'})
		#
		if create_session_ids:
			# Create sn_session_id
			# print('\tCreating Session IDs ...')
			df_compilation, sn_session_id_list = sn_session_id_maker(df_compilation)
			df_compilation, sn_test_id_list = sn_test_id_maker(df_compilation)
			print('\t\tSession ID creation complete.\n')

		# Adjust datetime if needed
		if len(datetime_fixme_dict) > 0:
			print("\tAdjusting datetime where necessary ...")
			for i in range(0, len(datetime_fixme_dict['Serial numbers'])):
				sn = datetime_fixme_dict['Serial numbers'][i]
				# isolate the manual timezone offset
				manual_timezone_offset_minutes = datetime_fixme_dict['timeoffsets_minutes'][i]
				manual_timezone_offset_minutes_utc = manual_timezone_offset_minutes
				# apply datetime offset to datetime_local in full dataframe
				df_compilation.loc[df_compilation['Serial number']== sn, 'datetime_local'] =\
					df_compilation.loc[df_compilation['Serial number']== sn, 'datetime_local'] +\
					pd.Timedelta(minutes = manual_timezone_offset_minutes)
				# apply datetime offset to datetime_utc in full dataframe
				df_compilation.loc[df_compilation['Serial number']== sn, 'datetime_utc'] =\
					df_compilation.loc[df_compilation['Serial number']== sn, 'datetime_utc'] +\
					pd.Timedelta(minutes = manual_timezone_offset_minutes_utc)

				# apply datetime offset to datetime_local in df_first_datapoint_per_file
				df_first_datapoint_per_file.loc[df_first_datapoint_per_file['Serial number']== sn, 'datetime_local'] =\
					df_first_datapoint_per_file.loc[df_first_datapoint_per_file['Serial number']== sn, 'datetime_local'] +\
					pd.Timedelta(minutes = manual_timezone_offset_minutes)
				# apply datetime offset to datetime_utc in df_first_datapoint_per_file
				df_first_datapoint_per_file.loc[df_first_datapoint_per_file['Serial number']== sn, 'datetime_utc'] =\
					df_first_datapoint_per_file.loc[df_first_datapoint_per_file['Serial number']== sn, 'datetime_utc'] +\
					pd.Timedelta(minutes = manual_timezone_offset_minutes_utc)

		if output_first_datapoint_of_each_file_separately:
			return df_compilation.reset_index(drop=True), df_first_datapoint_per_file
		else:
			return df_compilation.reset_index(drop=True)
