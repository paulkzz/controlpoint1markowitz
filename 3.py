# Загрузка данных по котировкам
import yfinance as yf
data = yf.download(['CL', 'MNST', 'MSFT', 'PG', 'AMZN'], period='12mo')

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

# Ежедневное изменение стоимости котировок
dCloseData = closeData.pct_change()
print(dCloseData)

# Средняя годовая доходность (252 рабочих дня)
returnMean = dCloseData.mean() * 252
print(returnMean)

# Ковариационная матрица
cov = dCloseData.cov()
print(cov)

# Формирование случайного портфеля
import numpy as np
cnt = len(dCloseData.columns)


def randPortf():
    res = np.exp(np.random.randn(cnt))
    res = res / res.sum()
    return res


r = randPortf()
print(r)
print(sum(r))


# Доходность портфеля
def returnPortf(r):
    return np.matmul(returnMean.values, r)


r = randPortf()
print(r)
d = returnPortf(r)
print(d)

# Риск портфеля (стандартная девиация)
def riskPortf(r):
    return np.sqrt(np.matmul(np.matmul(r, cov.values), r))


r = randPortf()
print(r)
rs = riskPortf(r)
print(rs)

# Создание 1000000 портфелей акций выбранных компаний
num_ports = 1000000

risk = np.zeros(num_ports)
ret = np.zeros(num_ports)
portf = np.zeros((num_ports, cnt))

for n in range(num_ports):
    r = randPortf()

    portf[n, :] = r
    risk[n] = riskPortf(r)
    ret[n] = returnPortf(r)

# Визуализация формируемых портфелей в виде облачной диаграммы
plt.figure(figsize=(10, 8))

plt.scatter(risk * 100, ret * 100, c='y', marker='.')
plt.xlabel('риск, %')
plt.ylabel('доходность, %')
plt.title("Облако портфелей")

# Портфель с минимальным риском
min_risk = np.argmin(risk)
plt.scatter([(risk[min_risk]) * 100], [(ret[min_risk]) * 100], c='r', marker='*', label='минимальный риск')

# Оптимальный портфель согласно коеффициенту Шарпа (отношения дохода к риску)
maxSharpKoef = np.argmax(ret / risk)
plt.scatter([risk[maxSharpKoef] * 100], [ret[maxSharpKoef] * 100], c='g', marker='o', label='максимальный коэф-т Шарпа')

# Портфель с максимальной доходностью
max_ret = np.argmax(ret)
plt.scatter([(risk[max_ret]) * 100], [ret[max_ret] * 100], c='b', marker='x', label='максимальная доходность')

plt.legend()

plt.show()

# Вывод данных о сформированных портфелях
import pandas as pd

print('--------------- Минимальный риск ---------------')
print("риск = %1.2f%%" % (float(risk[min_risk]) * 100.))
print("доходность = %1.2f%%" % (float(ret[min_risk]) * 100.))
print()
print(pd.DataFrame([portf[min_risk] * 100], columns=dCloseData.columns, index=['доли, %']).T)
print()

print('---------- Оптимальный портфель (наивысший коеффициент Шарпа) ----------')
print("риск = %1.2f%%" % (float(risk[maxSharpKoef]) * 100.))
print("доходность = %1.2f%%" % (float(ret[maxSharpKoef]) * 100.))
print()
print(pd.DataFrame([portf[maxSharpKoef] * 100], columns=dCloseData.columns, index=['доли, %']).T)
print()

print('--------------- Максимальная доходность ---------------')
print("риск = %1.2f%%" % (float(risk[max_ret]) * 100.))
print("доходность = %1.2f%%" % (float(ret[max_ret]) * 100.))
print()
print(pd.DataFrame([portf[max_ret] * 100], columns=dCloseData.columns, index=['доли, %']).T)
