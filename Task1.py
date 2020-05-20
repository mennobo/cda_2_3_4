# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import rcParams
import numpy as np
import pandas as pd
import seaborn as sns
plt.style.use('Solarize_Light2')
style.use('Solarize_Light2')
# %matplotlib inline
print(plt.style.available)
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
rcParams['figure.figsize'] = 6,5

df = pd.DataFrame(pd.read_csv('BATADAL_trainingset1.csv')) # No attacks
df_attacks = pd.DataFrame(pd.read_csv('BATADAL_trainingset2.csv')) # With attacks
df_nolabels = pd.DataFrame(pd.read_csv('BATADAL_test_dataset.csv')) # With attacks no labels
pd.set_option('display.expand_frame_repr', False)

# # Add missing attack labels
# The ATT_FLAG labels in dataset 2 in is incomplete, here we add the missing labels to the dataset.
# See https://batadal.net/images/Attacks_TrainingDataset2.png

df_attacks = df_attacks.set_index("DATETIME")
df_attacks[" ATT_FLAG"]["26/09/16 11":"27/09/16 10"] = 1 # Attack #2
df_attacks[" ATT_FLAG"]["29/10/16 19":"02/11/16 16"] = 1 # Attack #4
df_attacks[" ATT_FLAG"]["26/11/16 17":"29/11/16 04"] = 1 # Attack #5
df_attacks[" ATT_FLAG"]["06/12/16 07":"10/12/16 04"] = 1 # Attack #6
df_attacks[" ATT_FLAG"]["14/12/16 15":"19/12/16 04"] = 1 # Attack #7
df_attacks = df_attacks.reset_index()

# ## Task 1 - Familiarization

df.describe()


# +
data_preproc = pd.DataFrame({
    'date': df["DATETIME"],
    'F_PU1': df["F_PU1"],
    'F_PU2': df["F_PU2"],
    'F_PU4': df["F_PU4"],
    'F_PU7': df["F_PU7"],
})[1000:1500]
data_preproc2 = pd.DataFrame({
    'date': df["DATETIME"],
    'L_T1': df["L_T1"],
    'L_T3': df["L_T3"],
    'L_T5': df["L_T5"],
})[1000:1500]
data_preproc3 = pd.DataFrame({
    'date': df["DATETIME"],
    'P_J280': df["P_J280"],
    'P_J256': df["P_J256"],
    'P_J302': df["P_J302"],
    'P_J14': df["P_J14"],
})[1000:1500]

data_preproc.plot(figsize=(20,10), x='date')
data_preproc2.plot(figsize=(20,10), x='date')
data_preproc3.plot(figsize=(20,10), x='date')

# +
sns.heatmap(df.corr())
values = df['F_PU1']
plt.show()

# Remove all columns with a perfect correlation: 
perfect_cor = ['S_PU1', 'F_PU3', 'S_PU3', 'F_PU5', 'S_PU5', 'F_PU9', 'S_PU9', 'ATT_FLAG']
# check all the removed columns on their data (they all contain exactly the same value everywhere so they can be removed)
final_columns = list(df.columns)
for col in perfect_cor:
    print(df[col].value_counts())
    final_columns.remove(col)

def trimm_correlated(df_in, threshold):
    df_corr = df_in.corr(method='pearson', min_periods=1)
    df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)]*2, dtype=bool))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = df_in[un_corr_idx]
    return df_out

new_df = trimm_correlated(df[final_columns], 0.8)
sns.heatmap(new_df.corr())


# +
from numpy import mean
from sklearn.metrics import mean_squared_error

def moving_average_prediction(data, window = 3):
    test = [data[i] for i in range(window, len(data))]
    predictions = []
    
    current_prediction = window
    for t in range(len(test)):
        predicted_value = mean([data[i] for i in range(current_prediction-window,current_prediction)])
        predictions.append(predicted_value)
        current_prediction += 1
    # 	print('predicted=%f, expected=%f' % (yhat, obs))
    
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    return test, predictions


print('F_PU1 window 1')
data, predictions = moving_average_prediction(df['F_PU1'].values, 1)
print('F_PU1 window 2')
data, predictions = moving_average_prediction(df['F_PU1'].values, 2)
print('F_PU1 window 3')
data, predictions = moving_average_prediction(df['F_PU1'].values, 3)
print('F_PU1 window 4')
data, predictions = moving_average_prediction(df['F_PU1'].values, 4)

print('P_J14 window 1')
data, predictions = moving_average_prediction(df['P_J14'].values, 1)
print('P_J14 window 2')
data, predictions = moving_average_prediction(df['P_J14'].values, 2)
print('P_J14 window 3')
data, predictions = moving_average_prediction(df['P_J14'].values, 3)
print('P_J14 window 4')
data, predictions = moving_average_prediction(df['P_J14'].values, 4)

print('L_T1 window 1')
data, predictions = moving_average_prediction(df['L_T1'].values, 1)
print('L_T1 window 2')
data, predictions = moving_average_prediction(df['L_T1'].values, 2)
print('L_T1 window 3')
data, predictions = moving_average_prediction(df['L_T1'].values, 3)
print('L_T1 window 4')
data, predictions = moving_average_prediction(df['L_T1'].values, 4)

# plots
pd.DataFrame({"prediction":predictions[1000:2000],
            "actual": data[1000:2000]}).plot(figsize=(20,10))
# zoom plot
pd.DataFrame({"prediction":predictions[:100],
            "actual": data[:100]}).plot(figsize=(20,10))
# -

# # Task 2 - ARMA

# +
# %pip install scipy

# Make sure you have statsmodels >0.9.0 as it fails to import statsmodels.api
# see https://github.com/statsmodels/statsmodels/issues/5759
# %pip install git+https://github.com/statsmodels/statsmodels
    
# If the cell below this runs successfully you do NOT need this, especially the line 'import statsmodels.api as sm'

# +
import numpy as np
# from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

mpl.rcParams['figure.dpi'] = 150
rcParams['figure.figsize'] = 6,5

# -

# ## Autocorrelation function
# We calculate the autocorrelation and partial autocorrelation functions to make an informed descision about what ARMA parameters to use.

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
# from statsmodels.graphics.tsaplots import plot_acf
fig = sm.graphics.tsa.plot_acf(df['F_PU1'].values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['F_PU1'], lags=40, ax=ax2)

# +
# "The higher the AR order gets, the lower the AIC gets." you care about the rate of change. When the AIC does not drop substantially with the increase of an AR term, the search can stop for that sensor. 
def test_arma_params(train_series, params):
    # Find optimal parameters based on AIC 
    arma_mod = sm.tsa.ARMA(train_series, (0,0)).fit()
    
    zero_aic = arma_mod.aic
    best_params = params[0]
    lowest_aic = arma_mod.aic
    prev_aic = arma_mod.aic
    
    print(f"first aic is {prev_aic}")
    for param_set in params:
        print("testing " + str(param_set))
        try:
            arma_mod = sm.tsa.ARMA(train_series, param_set).fit()
            print(str(arma_mod.aic))
        except:
            continue
        print(f"Change: {arma_mod.aic - prev_aic}, change vs first: {arma_mod.aic - zero_aic}")
        prev_aic = arma_mod.aic
        if arma_mod.aic < lowest_aic:
            lowest_aic = arma_mod.aic
            best_params = param_set
            
    print('best params: ' + str(best_params))


def do_arma(train_series, test_series, params, attack_flags):
    print(f'####################################\nCurrent Series: {train_series.name}\n####################################')
    train_model = sm.tsa.ARMA(train_series, params).fit()#method='mle', trend='nc')
    test_model = sm.tsa.ARMA(test_series, params).fit(start_params = train_model.params)#, transpars = False, method='mle', trend='nc')

    #The equations are somewhat simpler if the time series is first reduced to zero-mean by subtracting the sample mean. Therefore, we will work with the mean-adjusted series

    # Plotting the residuals
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    resid = test_model.resid
    ax = resid.plot(ax=ax);

    # +
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    fig = qqplot(resid, line='q', ax=ax, fit=True)
    # -

    # ## ARMA Model Autocorrelation
    print("Autocorrelation plots:")
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)

    # ## Prediction
#     prediction = test_model.predict()
#     pd.DataFrame({"prediction":prediction[100:400],
#                 "actual": train_series[100:400]}).plot(figsize=(20,10))

    # ## Anomaly detection    
    resid = test_model.resid
    std = np.std(resid)
    anomaly_thresh = 2 * std
    detected_anomalies = test_model.resid[(resid) > anomaly_thresh]
    
    test_model = pd.DataFrame({ 'ATT_FLAG': attack_flags })
    tp=0
    fp=0
    for index, _ in detected_anomalies.items():
        if attack_flags[index]==1:
            tp+=1
        else:
            fp+=1
    tn=test_model.loc[attack_flags==-999].shape[0]-fp
    fn=test_model.loc[attack_flags==1].shape[0]-tp
    acc=100.0*(tp+tn)/(tp+tn+fp+fn)
    if (tp+fp)!=0:
        prec= 100.0 *tp / (tp + fp)
    else:
        prec=0
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    return detected_anomalies, resid


# +
def mean_absolute_err(y, yhat):
    return np.mean((np.abs(y.sub(yhat).mean()) / yhat)) # or percent error = * 100

def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()

def plot_attacks(residuals, attacks, detected_anomalies, show_range = (0,5000)):
    show_from = show_range[0]
    show_to = show_range[1]
    detected_attacks = []
    for a in range(len(df_attacks)):
            if a in detected_anomalies:
                detected_attacks.append(0.7)
            else:
                detected_attacks.append(-999)

    detected_attacks = pd.DataFrame(detected_attacks)
    plt.figure()
    residuals = residuals - np.mean(resid)
    plt.plot(residuals[show_from:show_to], label="residuals")
    plt.plot(attacks[show_from:show_to], label="Attacks")
    plt.plot(detected_attacks[show_from:show_to], label="Detected Attacks")

    axes = plt.gca()
    axes.set_ylim([np.min(residuals)*2,max(np.max(residuals)*1.5, 2)])
    plt.legend()
    plt.savefig("savedplot.png")
    plt.show()



# +
# param_sets = [(1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0)] # best 2,0
param_sets = [(2,0), (2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8)] # best 2,4
# test_arma_params(df['F_PU7'], param_sets)

anomalies, resid = do_arma(df['F_PU7'], df_attacks[' F_PU7'], (2,3), df_attacks[' ATT_FLAG'])
# Zoom in on especially Attack#5 and 6, which attacks F_PU7
plot_attacks(resid, df_attacks[' ATT_FLAG'], anomalies, (3400,3900))


# +
# param_sets = [(1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0)] # best 5,0
# param_sets = [(5,0), (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), (5,7), (5,8)] # best 5,2
# test_arma_params(df['L_T4'], param_sets)

anomalies, resid = do_arma(df['L_T4'], df_attacks[' L_T4'], (0,0), df_attacks[' ATT_FLAG'])
# Zoom in on especially Attack#5 and 6, which attacks F_PU7, affecting L_T4
plot_attacks(resid, df_attacks[' ATT_FLAG'], anomalies, (3000,4000))

# +
# param_sets = [(1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0)] # best 4,0
# param_sets = [(4,0), (4,1), (4,2), (4,3), (4, 4), (4,5), (4,6)] # best 4,2
# test_arma_params(df['L_T1'], param_sets)

anomalies, resid = do_arma(df['L_T1'], df_attacks[' L_T1'], (4,2), df_attacks[' ATT_FLAG'])
# zoom in on attacks 3 and 4 specifically
plot_attacks(resid, df_attacks[' ATT_FLAG'], anomalies, (1500,3000))


# +
# param_sets = [(1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0)] # best 3,0
# param_sets = [(3,0), (3,1), (3,2), (3,3), (3, 4), (3,5), (3,6)] # best 3,0 or 3,4
# test_arma_params(df['L_T7'], param_sets)

anomalies, resid = do_arma(df['L_T7'], df_attacks[' L_T7'], (3,0), df_attacks[' ATT_FLAG'])
plot_attacks(resid, df_attacks[' ATT_FLAG'], anomalies, (1500,3000))

# +
# param_sets = [(1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0)] # best 5,0
# param_sets = [(5,0), (5,1), (5,2), (5,3), (5,4), (5,5), (5,6)] # best 5,6
# test_arma_params(df['P_J300'], param_sets)

anomalies, resid = do_arma(df['P_J300'], df_attacks[' P_J300'], (5,6), df_attacks[' ATT_FLAG'])
plot_attacks(resid, df_attacks[' ATT_FLAG'], anomalies, (1500, 3000))
# + {}
# param_sets = [(1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0)] # best 2,0
# param_sets = [(2,0), (2,1), (2,2), (2,3), (2,4), (2,5), (2,6)] # best 2,2 or 2,5
# test_arma_params(df['F_PU10'], param_sets)

anomalies, resid = do_arma(df['F_PU10'], df_attacks[' F_PU10'], (2,5), df_attacks[' ATT_FLAG'])
plot_attacks(resid, df_attacks[' ATT_FLAG'], anomalies)
# -

# ## PCA Task

# +
# Preprocessing
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.decomposition import PCA

df = df.drop('DATETIME', axis=1)

def normalize(df):
    df_normalized = df.copy()
    df_normalized = df_normalized

    normalize = TimeSeriesScalerMeanVariance(mu=0, std=1)
    for col in df:
        df_normalized[col] = normalize.fit_transform(df_normalized[col])[0]

    return df_normalized

df_normalized = normalize(df)

# +
## Residuals 
pca = PCA()
pca.fit(df_normalized)
df_inverse_transformed = pca.inverse_transform(df_normalized)
pca_residual = df_normalized - df_inverse_transformed
pca_residual = np.square(pca_residual)
pca_residual_combined = pca_residual.sum(axis=1) 

figure, ax = plt.subplots()
plt.xlabel('Data points')
plt.ylabel('Residual')
plt.figure()
ax.plot(pca_residual_combined)
figure.savefig('pcaresidual.png')


# -

## Drop the abnormalities
indices_to_drop = np.where(pca_residual_combined > 2000)
print(indices_to_drop)
index = indices_to_drop[0]
print('before', df_normalized.shape)
df_cleaned = df_normalized.copy()
for index in indices_to_drop:
    row = df.iloc[index]
    df_cleaned = df_normalized.drop(row.index)
print('after', df_cleaned.shape)

# Re-normalize
df_cleaned_normalized = normalize(df_cleaned)


## Find importance of each principal component
pca = PCA()
pca.fit(df_cleaned_normalized)
x_axis = np.arange(1, df_cleaned_normalized.shape[1]+1, 1)
plt.xlabel('Principal Component')
plt.ylabel('Variance Captured')
plt.plot(x_axis, pca.explained_variance_ratio_)


# Cummulative Variance
cummulative_variance = pca.explained_variance_ratio_.cumsum()
x_axis = np.arange(1, df_cleaned_normalized.shape[1]+1, 1)
plt.xlabel('Principal components')
plt.ylabel('Cummulative variance captured')
plt.plot(x_axis, cummulative_variance)


# +
# Residual is now low
pca = PCA()
pca.fit(df_cleaned_normalized)
df_inverse_transformed = pca.inverse_transform(df_cleaned_normalized)
pca_residual = df_cleaned_normalized - df_inverse_transformed
pca_residual = np.square(pca_residual)
pca_residual_combined = pca_residual.sum(axis=1) 

figure, ax = plt.subplots()
plt.xlabel('Data points')
plt.ylabel('Residual')
plt.figure()
ax.plot(pca_residual_combined)
# -

# Prepare the test dataset
test_dataset = normalize(df_attacks.drop('DATETIME', axis=1).drop(' ATT_FLAG', axis=1))

# # Perform PCA analysis

# +
# Find threshold 
pca = PCA(n_components=15)
# pca.fit(df_cleaned_normalized)
transformed = pca.fit_transform(df_cleaned_normalized)
df_inverse_transformed = pca.inverse_transform(transformed)
pca_residual = df_cleaned_normalized - df_inverse_transformed
pca_residual = np.square(pca_residual)
pca_residual_combined = pca_residual.sum(axis=1) 
threshold_max = np.max(pca_residual_combined)
threshold_min = np.min(pca_residual_combined)

# analyse test set
pca = PCA(n_components=15)
pca.fit(test_dataset)
transformed = pca.fit_transform(test_dataset)
reconstructed = pca.inverse_transform(transformed)

residual_pca = test_dataset - reconstructed
residual_pca = np.square(residual_pca)
residual_pca = residual_pca.sum(axis=1) 

# Find attacks
attack_indices = np.where((residual_pca > threshold_max*2))
attack_indices2 = np.where((residual_pca < threshold_min*0.5))

all_detected_attacks = np.append(attack_indices[0], attack_indices2[0])

TP = 0
FP = 0
for index in all_detected_attacks:
    if index in list(df_attacks.loc[df_attacks[' ATT_FLAG']==1].index):
        TP +=1
    else:
        FP +=1 

print(f'TP={TP}\nFP={FP}')


# +
def plot_attacks(residuals, attacks, detected_anomalies):
    show_from = 0
    show_to = 5000
    detected_attacks = []
    for a in range(len(attacks)):
            if a in detected_anomalies:
                detected_attacks.append(0.5)
            else:
                detected_attacks.append(-99)

    detected_attacks = pd.DataFrame(detected_attacks)
    plt.figure(figsize=[10,5])
    residuals = residuals - np.mean(residuals)
#     plt.plot(residuals[show_from:show_to], label="residuals (normalized)")
    plt.plot(attacks[show_from:show_to], label="Actual attacks")
    plt.plot(detected_attacks[show_from:show_to], label="Detected Attacks")

    axes = plt.gca()
    axes.set_ylim([0,2])
    plt.legend()
    plt.savefig('pca_plot.png')
    plt.show()
    
plot_attacks(residual_pca, df_attacks[' ATT_FLAG'], all_detected_attacks)
# -
# # Discrete models

# +
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import SymbolicAggregateApproximation
from nltk import ngrams
from collections import Counter

def n_grams(n, data):
    gram = []
    for gr in ngrams(data, n):
        gram.append(''.join(gr))
    return gram

def perform_sax(dataset, gram_number, symbols, segments):
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=np.std(dataset))  # Rescale time series
    dataset = scaler.fit_transform(dataset)

    # SAX transform
    sax = SymbolicAggregateApproximation(n_segments=segments, alphabet_size_avg=symbols)
    sax_dataset_inv = sax.inverse_transform(sax.fit_transform(dataset))
    # print(pd.DataFrame(sax_dataset_inv[0])[0].value_counts())
#     sax_dataset_inv = sax.fit_transform(dataset)
#     print(len(sax_dataset_inv[0]))

    # Convert result to strings
    df_sax = pd.DataFrame(sax_dataset_inv[0])
    sax_series = df_sax[0]
    
    # Convert sax from numeric to characters
    sax_values = sax_series.unique()
    alphabet = 'abcdefghijklmnopqrstuvw'
    sax_dict = {x : alphabet[i]  for i, x in enumerate(sax_values)}
    sax_list = [sax_dict[x] for x in sax_series]
    
    # Convert the list of characters to n_grams based on input parameter
    tri = n_grams(gram_number, sax_list)
#     print(Counter(tri))
    return tri

def detect_anomaly(train, test, attacks, gram_number, symbols, segments):
    train_tri = perform_sax(train, gram_number, symbols, segments)
    test_tri = perform_sax(test, gram_number, symbols, segments)
#     print(train_tri)
#     print(test_tri)
    tp, fp, tn, fn = 0, 0, 0, 0
    anomaly_list = []
    for i, tri in enumerate(test_tri):
        attack = attacks[i]
        if tri in train_tri:
            if attack == 1:
                fn += 1
            else:
                tn += 1
        else:
            anomaly_list.append(i)
            if attack == 1:
                tp += 1
            else:
                fp += 1
    # Print scores
    if tp == 0 and fp == 0:
        return 'None', -1, list(), 0, 0
    else:
        precision = tp / (tp + fp)
        tag = f'experiment, symbols: {symbols}, segments: {segments}, gram_number: {gram_number}; fn: {fn}; tn: {tn}; fp: {fp}; tp: {tp}'
        return tag, precision, anomaly_list, fp, tp
            
def run_experiments(train, test, attack_indices):
    """
    Function that varies the parameters of the sax function
    """
    max_prec = 0
    max_tag = 'None'
    max_positives = 0
    for symbols in range(1, 20, 1):
        for segments in range(1, 250, 20):
            for gram_number in range(1, 6):
                    tag, precision, anom_list, fp, tp = detect_anomaly(train, test, attack_indices, gram_number, symbols, segments)
                    positives = tp + fp
                    if precision >= max_prec and positives > 20:
                        max_prec = precision
                        max_tag = tag
                        max_positives = positives
    return max_tag, max_prec

attack_indices = df_attacks[' ATT_FLAG']
for col in new_df.columns:
    print(col)
    max_tag, max_prec = run_experiments(new_df[col], df_attacks[' ' + col], attack_indices)
    print(f'Max prec: {max_prec}, tag: {max_tag}')


# +
def plot_attacks( attacks, detected_anomalies):
    show_from = 0
    show_to = 5000
    detected_attacks = []
    for a in range(len(attacks)):
            if a in detected_anomalies:
                detected_attacks.append(0.5)
            else:
                detected_attacks.append(-99)

    detected_attacks = pd.DataFrame(detected_attacks)
    plt.figure(figsize=[10,5])
    plt.plot(attacks[show_from:show_to], label="Actual attacks")
    plt.plot(detected_attacks[show_from:show_to], label="Detected Attacks")

    axes = plt.gca()
    axes.set_ylim([0,2])
    plt.legend()
    plt.savefig('pca_plot.png')
    plt.show()

attack_indices = df_attacks[' ATT_FLAG']
max_tag, max_prec, anomaly_list, fp, tp = detect_anomaly(new_df['L_T1'], df_attacks[' ' + 'L_T1'], attack_indices, 1, 5, 81)
print(max_tag)
plot_attacks(df_attacks[' ATT_FLAG'], anomaly_list)
