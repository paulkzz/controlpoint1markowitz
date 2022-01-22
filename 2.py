# Загрузка данных по котировкам
import yfinance as yf

data = yf.download(['AAPL', 'GE', 'BAC', 'AMD', 'PLUG', 'F'], period='12mo')

# Курсы закрытия
closeData = data.Close
print(closeData)

# Графики курсов
import matplotlib.pyplot as plt

for name in closeData.columns:
    closeData[name].plot()
    plt.grid()
    plt.title(name)
    plt.show()

# Изменение курсов
dCloseData = closeData.pct_change()
print(dCloseData)

# Средняя доходность
dohMean = dCloseData.mean()
print(dohMean)

# Ковариация
cov = dCloseData.cov()
print(cov)

# Случайный портфель
import numpy as np

cnt = len(dCloseData.columns)


def randPortf():
    res = np.exp(np.random.randn(cnt))
    res = res / res.sum()
    return res


r = randPortf()
print(r)
print(r.sum())


# Доходность портфеля
def dohPortf(r):
    return np.matmul(dohMean.values, r)


r = randPortf()
print(r)
d = dohPortf(r)
print(d)


# Риск портфеля
def riskPortf(r):
    return np.sqrt(np.matmul(np.matmul(r, cov.values), r))


r = randPortf()
print(r)
rs = riskPortf(r)
print(rs)

# Облако портфелей
risk = np.zeros(N)
doh = np.zeros(N)
portf = np.zeros((N, cnt))

for n in range(N):
    r = randPortf()

    portf[n, :] = r
    risk[n] = riskPortf(r)
    doh[n] = dohPortf(r)

plt.figure(figsize=(10, 8))

plt.scatter(risk * 100, doh * 100, c='y', marker='.')
plt.xlabel('риск, %')
plt.ylabel('доходность, %')
plt.title("Облако портфелей")

min_risk = np.argmin(risk)
plt.scatter([(risk[min_risk]) * 100], [(doh[min_risk]) * 100], c='r', marker='*', label='минимальный риск')

maxSharpKoef = np.argmax(doh / risk)
plt.scatter([risk[maxSharpKoef] * 100], [doh[maxSharpKoef] * 100], c='g', marker='o', label='максимальный коэф-т Шарпа')

r_mean = np.ones(cnt) / cnt
risk_mean = riskPortf(r_mean)
doh_mean = dohPortf(r_mean)
plt.scatter([risk_mean * 100], [doh_mean * 100], c='b', marker='x', label='усредненный портфель')

plt.legend()

plt.show()

# Выведем данные найденных портфелей.
import pandas as pd

print('---------- Минимальный риск ----------')
print()
print("риск = %1.2f%%" % (float(risk[min_risk]) * 100.))
print("доходность = %1.2f%%" % (float(doh[min_risk]) * 100.))
print()
print(pd.DataFrame([portf[min_risk] * 100], columns=dCloseData.columns, index=['доли, %']).T)
print()

print('---------- Максимальный коэффициент Шарпа ----------')
print()
print("риск = %1.2f%%" % (float(risk[maxSharpKoef]) * 100.))
print("доходность = %1.2f%%" % (float(doh[maxSharpKoef]) * 100.))
print()
print(pd.DataFrame([portf[maxSharpKoef] * 100], columns=dCloseData.columns, index=['доли, %']).T)
print()

print('---------- Средний портфель ----------')
print()
print("риск = %1.2f%%" % (float(risk_mean) * 100.))
print("доходность = %1.2f%%" % (float(doh_mean) * 100.))
print()
print(pd.DataFrame([r_mean * 100], columns=dCloseData.columns, index=['доли, %']).T)
print()