import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.plotting import autocorrelation_plot

# 정상성
## 일정한 평균
## 일정한 분산
## 일정한 자기상관
## 주기 성분 없음

## seed 값을 가지고 난수를 생성한다.
np.random.seed(101)

time = np.arange(500)
stationary = np.random.normal(loc = 0, scale = 1.0, size = len(time))

def plot_sequence(x, y, title):
    plt.figure(figsize=(16, 4))
    plt.plot(x, y)
    plt.xlabel('time')
    plt.ylabel('series value')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()

plot_sequence(time, stationary, 'Stationary se ries')

pd.Series(stationary).hist()
plt.show()


# non-stationary한 데이터
seed = 3.14
# 메모리에 남은 아무 값이 표현해준다.
lagged = np.empty_like(time, dtype = 'float32')
for t in time:
    lagged[t] = seed + np.random.normal(loc = 0, scale = 2.5, size = 1)
    seed = lagged[t]

plot_sequence(time, lagged, 'Non-stationary Time Series = Lagged Structure')
plt.show()

pd.Series(lagged).hist()
plt.show()

# Trend한 데이터
trend = time + stationary * 50
plot_sequence(time, trend, 'Non-stationary - Trend')
plt.show()

# 이분산성(변동 분산)
np.random.seed(123)
level_1 = np.random.normal(loc = 0, scale = 1.0, size = 250)
level_2 = np.random.normal(loc = 0, scale = 10.0, size = 250)
data = np.append(level_1, level_2)
print(data)
print(data.shape)
plot_sequence(time, data, 'Non-stationary - Varying Variance')

seasonality = 10 + np.sin(time) * 10
plot_sequence(time, seasonality, 'Non-stationary - Seasonality')
plt.show()

trend_seasonality = trend + seasonality
plot_sequence(time, trend_seasonality, 'Non-stationary - Trend+Seasonality')
plt.show()

auto_correlation = lagged * 10 + trend
plot_sequence(time, auto_correlation, 'Non-stationary - Autocorrelation + Trend')
plt.show()

time2 = np.arange(time.size * 2)
trend2 = -time + stationary * 50
financial_crisis = np.append(trend, trend2)

plot_sequence(time2, financial_crisis, 'Non-stationary - Financial Crisis')
plt.show()

# Differencing 으로 Autocorrelation 제거
#
difference = lagged[:-1] - lagged[1:]
plot_sequence(time[:-1], difference, 'stationary Time Series - difference')
plt.show()

pd.Series(difference).hist()
plt.show()

# pandas autocorrelation_plot
autocorrelation_plot(stationary)
plt.show()

autocorrelation_plot(lagged)
plt.show()

autocorrelation_plot(financial_crisis)
plt.show()

autocorrelation_plot(trend_seasonality)
plt.show()
