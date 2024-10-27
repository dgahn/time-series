import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# 자기상관성이란 시계열 데이터에서 시간에 따른 데이터값들의 상관관계를 의미한다.
# 현재 시점의 값이 이전 시점의 값과 얼마나 연관되어 있는지를 타나내는 개념이다.
# 자기상관성은 주로 시간의 흐름에 따라 발생하는 규칙성을 확인하고 패턴을 분석할 때 유용하다.
#
# 지연 시간을 기준으로 계산하는데 예를 들어, lag=1일 때 자기상관성은 현재 값과 바로 이전 값 간의 상관관계를 의미한다.
# 자기상관성이 존재하면, 시계열 데이터가 과거 데이터의 영향을 받아 일정한 패턴이나 트렌드를 유지하고 있다는 뜻이다.
#
# 양의 자기상관성: 상관계수가 양의 값일 경우, 자기상관성이 양의 값이라는 것은 트렌드가 일정하다는 것을 의미한다. 과거 값이 커지면 현재의 값이 커지고 과거의 값이 작아지면 현재의 값이 작아진다.
# 두 값의 간의 관계가 정비례 관계에 있다는 것을 의미한다.
# 음의 자기상관성: 상관계수가 음의 값일 경우, 과거 값이 커질수록 현재 값은 작아질 확률이 높다는 것을 의미한다. lag 사이에 있는 두 값이 정비례 관계에 있다는 것을 의미한다.

# 자기상관성의 활용
# 예측 모델링: 자기상관성이 존재하는 시계열 데이터는 과거의 값이 현재나 미래의 값을 예측하는데 도움을 줄 수 있다. 예를 드렁, AR(자기회귀) 모델은 자기상관성을 이용하여 미래의 값을 예측한다.
# 패턴 감지: 자기상관성을 분석함으로써 데이터가 일정한 주기성을 가지는지, 또는 랜덤한 변동성을 가지는지 파악할 수 있다.

# 데이터를 분석하여 어떤 패턴인지 파악하는게 중요.
# 패턴에 따라 어떤 알고리즘 적용할지 선택해야한다.


# 태양 흑점 data의 계절성, 자기 상관성 분석
path = './datasets/Sunspots.csv'
df = pd.read_csv(path, skiprows=0, index_col=0)
print(df.head())
## df.info()를 통해 각 열이 어떤 값을 가지고 있는지 확인할 수 있다.
print(df.info())
## object로 되어 있던 date를 datetime 타입으로 변경한다.
df['Date'] = pd.to_datetime(df['Date'])
## index를 Date 열로 변경한다.
df.set_index('Date', inplace=True)
print(df.info())
print(df.head())

## asfreq() 함수는 시계열 데이터의 빈도를 조정하는데 사용되는 pandas 라이브러리의 함수다.
## 월별 데이터 빈도, 일별 데이터 빈도 등으로 변환할 수 있는 특징이 있다.
series = df["Monthly Mean Total Sunspot Number"].asfreq("ME")
series.plot(figsize=(8, 4))
plt.show()

# 1995년 1월 1일부터의 데이터를 포함하는 것은 아래와 같이 작성한다.
series["1995-01-01":].plot()
plt.show()

from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt

plt.figure(figsize = (12, 2))
autocorrelation_plot(series)
plt.title("Autocorrelation Plot", fontsize = 14) # 그래프 제목 추가
plt.xlabel("Lag", fontsize = 12)
plt.ylabel("Autocorrelation", fontsize = 12)
plt.show()

# 자기상관성 함수의 lag=0에서의 값은 항상 1.0을 가진다. 이는 첫값은 자기상관성을 자기와 가지기 때문
# 이후에는 이전 시점과의 상관관계를 나타내며, 시간이 지남에 따라 감소하거나 증가할 수 있다.
# lag=1은 현재 값과 한 시점 전의 값의 상관관계를 의미한다.
# 즉, 그래프에서 autocorrelation 값이 1.0에 가까운 값을 보여주기 때문에 한시점 값은 비슷한 값을 가진다고 할 수 있다.
# lag의 값이 1.0이거나 -1.0인 경우에 현 시점으로부터 lag 시점의 값이 영향을 받는다는 것을 의미하고 0에 가까워질수록 관계가 없다는 것이다.
# 그러므로 0으로 수렴한다는 것은 과거의 데이터는 의미가 없다는 것을 의미한다.
# 그리고 해석할 수 있는 것은 긴주기성보다는 짧은 주기성을 가진 값이라고 본다.

# 그래프를 보면 Autocorrelation의 값이 양의 피크의 값에 갔다가 음의 피크의 값으로 가는데 이는 현재 시점에 비해 값이 비슷해졌다가
# 반대로 적어졌다는 것을 알 수 있다. 그 의미는 주기적으로 흑점의 수가 변화 했다는 것을 의미한다.

# ADF 검정부터 다시 시작.