import pandas as pd
import numpy as np

df = pd.read_excel('./date/datasets/superstore.xls')
info = df.info
print(info)

# 앞에 5개
head = df.head()
print(head)

# 뒤에 5개
tail = df.tail()
print(tail)

## 판다스는 numpy를 가지고 만들었다.

## groupby는 집계연산을 할 수 있다.
group_variables = ['Order Date', 'Category']
grouped_df = df.groupby(group_variables)

## 'Sales'를 기준으로 묶어서 sum을 한 결과를 보여준다.
## 열에 index를 추가해서 보여준다.
sales = df.groupby(group_variables)['Sales'].sum().reset_index()
print(sales.head())

# Sales의 데이터 타입을 알려준다.
print(sales.dtypes)

order_date = sales['Order Date'].values
print(order_date)

order_date_daily = np.array(order_date, dtype='datetime64[D]')
print(order_date_daily)

order_date_monthly = np.array(order_date, dtype='datetime64[M]')
print(order_date_monthly)

print(np.unique(order_date_monthly))
print(np.unique(np.array(order_date, dtype='datetime64[Y]')))

# 크기가 (12, 4)인 임의의 표준 정규 분포에 따른 난수 배열을 생성한다.
a = np.random.standard_normal((12, 4))
df = pd.DataFrame(a, columns=['n1', 'n2', 'n3', 'n4'])
df.head()
print(df)

# datetime 자료형으로 변경해준다.
date_str = ['2010-01-01', '2015, 7, 1', 'May, 1 2016',
            'Dec, 25, 2019', 'DEC 1 2020', 'dec 20 2021', '2020 12 31']
for dt in date_str:
    print(pd.to_datetime(dt))

# 시작일과 기간을 통해서 날짜 범위를 생성할 수 있다.
range1 = pd.date_range(start='2016-9-1', periods = 10)
print(range1)

# 시작일과 종료일, 그리고 freq인자('B' 지정시 평일로 지정)를 통해 평일 범위를 만들 수 있다.
range2 = pd.date_range(start='2019-5-1', end='2019-5-10', freq='B')
print(range2)

# 시작일과 기간, 그리고 freq인자('ME' 지정시 월 마지막 날짜를 기준)을 통해 월 마지막 날짜 범위를 생성한다.
range3 = pd.date_range('2019-01-01', periods=12, freq='ME')
print(range3)

# DateFrame의 index를 datetime으로 변경
datedf = pd.DataFrame(
    np.random.standard_normal((12, 4)),
    columns=['n1', 'n2', 'n3', 'n4'],
)
datedf.index = range3
print(datedf)
print(datedf.index)
print(1)
print(sales)
# index를 Order Date로 변경한다. inplace를 True로 지정하는 경우 새로운 객체로 만드는 것이 아니라 기존 객체를 수정한다.
sales.set_index('Order Date', inplace=True)
print(2)
print(sales.head())

# sales의 index를 출력하려면 아래와 같이 하면 된다.
print(sales.index)

# 2011년에 해당하는 데이터만 선택한다.
print(sales.loc['2011'].head())

# 'Category'가 'Office Supplies'인 데이터 중, 2012년 1월 - 2월인 데이터를 선택
print(sales[sales['Category'] == 'Office Supplies']['2012-01':'2012-02'].head())

# 인덱스에서 일(day) 정보를 가져오는 예제
print(sales.index.day)

# 인덱스에서 요일(day of week) 정보를 가져오는 예제
print(sales.index.dayofweek)

# Series는 1차원 배열이지만 인덱스를 추가하는 특징이 있다. 그래서 인덱스를 앞에 선언한다. 기본적은 series는 어떤 데이터를 넣을지만 정의하면 되지만
# 아래는 인덱스로 어떤 것을 사용할지 지정하였다.
ts = pd.Series(np.random.randn(100), index=pd.date_range('2018-1-1', periods=100, freq="ME"))
print(ts)

print(ts.tail())

# 주단위로 다운 샘플링을 할 수 있다.
print(f"resample: \n{ts.resample('W').mean()}")
# 월단위로 다운 샘플링을 할 수 있다.
print(f"resample: \n{ts.resample('ME').mean()}")
