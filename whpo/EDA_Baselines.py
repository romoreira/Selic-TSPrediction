# %%
"""
This notebook is used for exploratory data analysis.
"""

# %%
import warnings

warnings.filterwarnings('ignore')

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# %%
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose

# %%
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA


from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_error

# %%
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

# %%
large = 22;
med = 16;
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (10, 6),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")


# Version
print(mpl.__version__)
print(sns.__version__)

# %%
file_name = "selicdados2.csv"

history = 24  # input historical time steps
horizon = 1  # output predicted time steps
test_ratio = 0.2  # testing data ratio
max_evals = 50  # maximal trials for hyper parameter tuning

# Save the results
#y_true_fn = '%s_true-%d-%d.pkl' % (model_name, history, horizon)
#y_pred_fn = '%s_pred-%d-%d.pkl' % (model_name, history, horizon)
df = pd.read_csv(file_name, sep=',', index_col=[0], parse_dates=True)
print(df.head())

# divide data into train and test
train_ind = int(len(df) * 0.8)
train = df[:train_ind]
test = df[train_ind:]
print(train.head())
print(test.head())
train_length = train.shape[0]
test_length = test.shape[0]

print('Training size: ', train_length)
print('Test size: ', test_length)
print('Test ratio: ', test_length / (test_length + train_length))

# Dickey Fuller Test
adfinput = adfuller(train['SelicDia'])
adftest = pd.Series(adfinput[0:4], index=['Dickey Fuller Statistical Test', 'P-value',
                                          'Used Lags', 'Number of comments used'])
adftest = round(adftest, 4)

for key, value in adfinput[4].items():
    adftest["Critical Value (%s)" % key] = value.round(4)
adftest

kpss_input = kpss(train['SelicDia'])
kpss_test = pd.Series(kpss_input[0:3], index=['Statistical Test KPSS', 'P-Value', 'Used Lags'])
kpss_test = round(kpss_test, 4)

for key, value in kpss_input[3].items():
    kpss_test["Critical Value (%s)" % key] = value
kpss_test

plot_acf(train['SelicDia'], lags=24 * 7, zero=False);

plot_pacf(train['SelicDia'], lags=24 * 7, zero=False);


def check_error(orig, pred, name_col='', index_name=''):
    bias = np.mean(orig - pred)
    mse = mean_squared_error(orig, pred)
    rmse = sqrt(mean_squared_error(orig, pred))
    mae = mean_absolute_error(orig, pred)
    mape = np.mean(np.abs((orig - pred) / orig)) * 100

    error_group = [bias, mse, rmse, mae, mape]
    result = pd.DataFrame(error_group, index=['BIAS', 'MSE', 'RMSE', 'MAE', 'MAPE'], columns=[name_col])
    result.index.name = index_name
    print(str(result))
    return result


def plot_error(data, figsize=(12, 9), lags=24, rotation=0):
    # Creating the column error
    data['Error'] = data.iloc[:, 0] - data.iloc[:, 1]

    plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax4 = plt.subplot2grid((2, 2), (1, 1))

    # Plotting actual and predicted values
    ax1.plot(data.iloc[:, 0:2])
    ax1.legend(['Real', 'Pred'])
    ax1.set_title('Real Value vs Prediction')
    ax1.xaxis.set_tick_params(rotation=rotation)

    # Error vs Predicted value
    ax2.scatter(data.iloc[:, 1], data.iloc[:, 2])
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residual')
    ax2.set_title('Residual vs Predicted Values')

    # Residual QQ Plot
    sm.graphics.qqplot(data.iloc[:, 2], line='r', ax=ax3)

    # Autocorrelation Plot of residual
    plot_acf(data.iloc[:, 2], lags=lags, zero=False, ax=ax4)
    plt.tight_layout()
    plt.show()


target = 'SelicDia'



# %%
# C reating the training variable to compare with the error later
simple_train = train[[target]]
simple_train.columns = ['Real']
simple_train['Pred'] = simple_train['Real'].shift()
simple_train.dropna(inplace=True)

# %%
simple_train.head()

# %%
"""
Let's create a variable to check the training error of this model, we will also plot the graphs mentioned above:
"""

# %%
error_train = check_error(simple_train['Real'],
                          simple_train['Pred'],
                          name_col='Simple',
                          index_name='Training Base')

print('SIMPLE MODEL IN THE TRAINING DATA')
plot_error(simple_train)
error_train


simple_test = test[[target]]
simple_test.columns = ['Real']

# adding the first value of the Forecast with the last Actual data of the test
hist = [simple_train.iloc[i, 0] for i in range(len(simple_train))]

pred = []
for t in range(len(simple_test)):
    yhat = hist[-1]
    obs = simple_test.iloc[t, 0]
    pred.append(yhat)
    hist.append(obs)

simple_test['Pred'] = pred

# creating the basis of error in the test
error_test = check_error(simple_test['Real'],
                         simple_test['Pred'],
                         name_col='Simple',
                         index_name='Testing Base')

print('SIMPLE MODEL IN THE TEST DATA')
plot_error(simple_test, rotation=45)
error_test

# %%
"""
# Simple Moving Average
"""

# %%
"""
The moving average is an average that is calculated for a given period (5 hours for example) and is moving and always being calculated using this particular period.
"""

# %%
sma_train = train[[target]]
sma_train.columns = ['Real']
sma_train['Pred'] = sma_train.rolling(5).mean()
sma_train.dropna(inplace=True)

# Checking the error of the moving averages on the training model
error_train['5H Moving Avg'] = check_error(sma_train['Real'], sma_train['Pred'])
error_train

# %%
"""
The error is above the simple model.
"""

# %%
sma_test = test[[target]]
sma_test.columns = ['Real']

# Continuing to use the 5-hour moving average step by step:
hist = [sma_train.iloc[i, 0] for i in range(len(sma_train))]
pred = []
for t in range(len(sma_test)):
    yhat = np.mean(hist[-5:])
    obs = sma_test.iloc[t, 0]
    pred.append(yhat)
    hist.append(obs)

sma_test['Pred'] = pred

# plotting the test chart
print('5-Hour MOVING AVERAGE MODEL ON THE TEST DATA')
plot_error(sma_test, rotation=45)

# Checking the error of the moving average on test model
error_test['5H Moving Avg'] = check_error(sma_test['Real'], sma_test['Pred'])


# %%
"""
The test error is also above the simple model.
"""

# %%
"""
# Exponential Moving Average
"""

# %%
"""
$\alpha$(alpha) is a constant with a value between 0 and 1, we will calculate the forecast with the following formula:
"""

# %%
"""
$$Ypred_t=Ypred_{t−1}+\alpha(Y_{t−1}−Ypred_{t−1})$$
"""

# %%
"""
We try different alpha values:
"""

# %%
#emm = train[[target]]
#alpha_ = [0, 0.2, 1]
#for key, value in enumerate(alpha_):
#    model = ExponentialSmoothing(emm[target]).fit(smoothing_level=value)
#    emm[f'Alpha {value}'] = model.predict(start=0, end=len(emm) - 1)

# plotting part of the graph to improve visualization
#emm[:20].plot(figsize=(12, 9), title='Multiple alpha values versus training series')
#plt.show()



emm_train = train[[target]]
emm_train.columns = ['Real']

# Creating the model:
alpha = 0.2
model = ExponentialSmoothing(emm_train['Real']).fit(smoothing_level=alpha)
emm_train['Pred'] = model.predict(start=0, end=len(emm_train) - 1)

# Checking the error of the exponential moving averages training model
error_train['Exp. Moving Avg'] = check_error(emm_train['Real'], emm_train['Pred'])
error_train

# %%
"""
The error is better than the simple moving average, but still above the simple method.
"""

# %%
emm_test = test[[target]]
emm_test.columns = ['Real']

# creating the model
hist = [emm_train.iloc[i, 0] for i in range(len(emm_train))]
hist_pred = [emm_train.iloc[i, 1] for i in range(len(emm_train))]
pred = []
for t in range(len(emm_test)):
    yhat = hist_pred[-1] + alpha * (hist[-1] - hist_pred[-1])
    obs = emm_test.iloc[t, 0]
    pred.append(yhat)
    hist.append(obs)
    hist_pred.append(yhat)

emm_test['Pred'] = pred

# plotting the test chart
print('EXPONENTIAL MOVING AVERAGE WITH 0.50 ALPHA ON THE TEST DATA')
plot_error(emm_test, rotation=45)

# Checking the error of the exponential moving averages test model
error_test['Exp. Moving Avg'] = check_error(emm_test['Real'], emm_test['Pred'])
print("Exp. Moving Avg: "+str(error_test))

# %%
ar_train = train[[target]]
ar_train.columns = ['Real']

# Creating the model:
# use 2 lags
model = ARIMA(ar_train['Real'], order=[2, 0, 0]).fit()
ar_train['Pred'] = model.predict(start=0, end=len(ar_train) - 1)


def f_zero(x):
    if x > 0:
        return x
    else:
        return 0


ar_train['Pred'] = ar_train['Pred'].apply(f_zero)

# Checking the auto regressive model error
error_train['Auto Regr.'] = check_error(ar_train['Real'], ar_train['Pred'])
print("Auto Regr: "+str(error_train))


# %%
ar_test = test[[target]]
ar_test.columns = ['Real']

# validating the data using the coefficients of the trained model
coef_l1, coef_l2 = model.arparams
hist = [ar_train.iloc[i, 0] for i in range(len(ar_train))]
pred = []
for t in range(len(ar_test)):
    yhat = (hist[-1] * coef_l1) + (hist[-2] * coef_l2)
    obs = ar_test.iloc[t, 0]
    pred.append(yhat)
    hist.append(obs)

ar_test['Pred'] = pred

ar_test['Pred'] = ar_test['Pred'].apply(f_zero)

# plotting the test chart
print('AUTO REGRESSIVE MODEL IN THE TEST DATA')
plot_error(ar_test, rotation=45)

# Checking the auto regressive model error
error_test['Auto Regr.'] = check_error(ar_test['Real'], ar_test['Pred'])
error_test



# %%
input_features = ['SelicDia']
target = 'SelicDia'


# %%
maxlag = 24
_test = 'ssr_chi2test'


def grangers_causation_matrix(data, variables, _test='ssr_chi2test', verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][_test][1], 4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


grangers_causation_matrix(train, variables=input_features)


def cointegration_test(df, alpha=0.05):
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df, -1, 5)
    d = {'0.90': 0, '0.95': 1, '0.99': 2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1 - alpha)]]

    def adjust(val, length=6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--' * 20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace, 2), 9), ">", adjust(cvt, 8), ' =>  ', trace > cvt)


cointegration_test(train[input_features])

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
    p_value = output['pvalue']

    def adjust(val, length=6):
        return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-' * 47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key, val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")

    # %%

# %%
# ADF Test on each column
for name, column in train[input_features].iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

exit()#Para aqui porque temos apenas uma unica variável
data = train[input_features]
model = VAR(data)
results = model.select_order(maxlags=15)
results.summary()
model_fitted = model.fit(24)
model_fitted.summary()
lag_order = model_fitted.k_ar
lag_order
var_train = train[[target]]
var_train.columns = ['Real']
input_data = train[input_features].values
predicted_values = []
for i in range(input_data.shape[0]):
    if i < lag_order:
        predicted_values.append(np.nan)
    else:
        input_batch = input_data[i - lag_order:i]
        result = model_fitted.forecast(y=input_batch, steps=1)
        predicted_values.append(result[0][0])
var_train['Pred'] = predicted_values
var_train = var_train.dropna()
var_train['Pred'] = var_train['Pred'].apply(f_zero)
error_train['VAR'] = check_error(var_train['Real'], var_train['Pred'])
error_train
var_test = test[[target]]
var_test.columns = ['Real']
train_data = train[input_features].values
test_data = test[input_features].values
hist = train_data
pred = []
for t in range(len(var_test)):
    input_batch = hist[-lag_order:]
    result = model_fitted.forecast(y=input_batch, steps=1)
    yhat = result[0][0]
    obs = test_data[t]
    pred.append(yhat)
    hist = np.vstack([hist, obs])
var_test['Pred'] = pred
var_test['Pred'] = var_test['Pred'].apply(f_zero)
print('VAR MODEL IN THE TEST DATA')
plot_error(var_test, rotation=45)
error_test['VAR'] = check_error(var_test['Real'], var_test['Pred'])
error_test
model_name = 'VAR'
history = 24
horizon = 1
y_pred_fn = '%s_pred-%d-%d.pkl' % (model_name, history, horizon)
import pickle
pred = np.array(var_test['Pred'])
pickle.dump(pred, open(y_pred_fn, 'wb'))