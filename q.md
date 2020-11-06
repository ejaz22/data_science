# Time series analysis / Time series forecasting methods
Time series and text are both sequential data.
classical algorithms, ML Based algorithms, Probabilistic models
Stationarity ( Constant mean, constant variance and autocovariance that does not depend on time)
Copulas and Baysian infrernce approaches
regression approaches vs time series methods 
Assumption of Regression Methods - patterns in the past data will be repeated in future.
we cannot use a conventional cross validation approach,we have to split a historical data set on the training set and validation set by using period splitting, so the training data will lie in the first time period and the validation set in the next one.

Trend in time-series is a pattern(positve or negative, linear or non-linear). No trend implies implies it is stationary i.e. data has constant mean and variance over time.
Seasonality recur in regular intervals and is predictable whereas cyclicality recur in irregular interval.
A time series is stationarity if it does not exhibit any trend or seasonality.
White noise is a time series that is purely random in nature. Mean and variance of the same is always constant. If data is a white noise then intelligent forecasting is not  possible. In the case mean is the prediction.

Simple moving average captures trend but poor at capturing other components.
Exponential Smoothing (EWMA) gives high weight to nearby values and low weights to far off values. Seasonality is better captured in EWMA as compared to MA.
Autoregressive (AR), if series is not white noise then we can use it.



### Models
- Simple Moving Average (Rolling mean)
- Exponential Smoothing
- Double Exponential Smoothing 
- Triple Exponential Smoothing aka Holt-Winters
- Holts exponential smoothing : a TS forecasting method for univariate data
- Holt Winters additive model (HWAAS) : Exponential smoothing with additive trend and additive seasonality
- TBAT (Trigonometric Seasonal Formulation, Box-Cox transformation, ARMA errors, Trend Component)
- DeepAR : Probabilistic Forecasting with Auto-Regressive Recurrent Networks
- FBProphet : Automatic forecasting procedure
- ARIMA
- SARIMA
- SARIMAX
- GARCH
- N-Beats : Neural Basis Expansion Analysis for Interpretable Time Series Forecasting
- Gluonts
