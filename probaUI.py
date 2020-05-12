import statsmodels.api as sm
import pylab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog as fd
import PIL as pl
from PIL import Image, ImageTk
from scipy import stats
from itertools import product
import warnings
from tkinter import ttk
import time
from datetime import *
from dateutil.relativedelta import relativedelta
c="Критерий Дики-Фуллера имеет следующий показатель: "
file_name = ''



# Дифференцирование
def differents():
    clear_root()
    global period
    global count_of_diff
    global count_of_season_diff
    clear_root()
    wine['sales_box'] = wine.sales_box - wine.sales_box.shift(1)
    if (sm.tsa.stattools.adfuller(wine.sales_box[1:])[1] > 0.001 or d[1]<=0.1):
        wine['sales_box']= wine.sales_box - wine.sales_box.shift(period)
        print(wine)
        count_of_season_diff+=1
        plt.figure(figsize=(15, 7))
        wine.sales_box[(1+period):].plot()
        plt.ylabel('value')
        pylab.savefig('graf2.png')
        scale_image('graf2.png', 'graf2_1.png', window_size_X)
        img = ImageTk.PhotoImage(Image.open("graf2_1.png"))
        l3.config(image=img)
        l3.photo_ref = img
        b=sm.tsa.stattools.adfuller(wine.sales_box[(1+period*count_of_season_diff):])[1]
        global c
        d=c+str(b)
        kr_dik_fuler["text"]=d
        l3.grid_forget()
        kr_dik_fuler.grid()
        if (b>0.01):
            stationary_ts["text"] = "Ряд не стационарен, необходимы преобразования"
            stationary_ts.grid()
            l9.grid()
            btn6["command"]=differents()
            btn6.grid()
        elif(b<=0.01 and b>=0):
            stationary_ts["text"] = "Ряд стационарен, возможно прогнозирование"
            stationary_ts.grid()
            l10.grid()
            l3.grid()
            l12.grid()
            btn5.grid()
    else:
        plt.figure(figsize=(15, 7))
        wine.sales_box[(1):].plot()
        plt.ylabel('value')
        pylab.savefig('graf2.png')
        scale_image('graf2.png', 'graf2_1.png', window_size_X)
        img = ImageTk.PhotoImage(Image.open("graf2_1.png"))
        l3.config(image=img)
        l3.photo_ref = img
        stationary_ts["text"] = "Ряд стационарен, возможно прогнозирование"
        stationary_ts.grid()
        l10.grid()
        l3.grid()
        l12.grid()
        btn5.grid()



# Прогноз


def prognoz():
    clear_root()
    global date_list
    global length_mass
    wine2 = wine[['val']]
    try:
        if (wine.index.freq == "D"):
            date_list = [datetime.strptime(str(wine.index[len(wine.index) - 1].date()),'%Y-%m-%d') + relativedelta(days=x) for x in range(0, 30)]
        elif(wine.index.freq == "W"):
            date_list = [datetime.strptime(str(wine.index[len(wine.index) - 1].date()),'%Y-%m-%d') + relativedelta(weeks=x) for x in range(0, 30)]
        elif(wine.index.freq == "MS"):
            date_list = [datetime.strptime(str(wine.index[len(wine.index) - 1].date()),'%Y-%m-%d') + relativedelta(months=x) for x in range(0, 30)]
        elif(wine.index.freq == "M"):
            date_list = [datetime.strptime(str(wine.index[len(wine.index) - 1].date()),'%Y-%m-%d') + relativedelta(months=x) for x in range(0, 30)]
        elif(wine.index.freq == "Y"):
            date_list = [datetime.strptime(str(wine.index[len(wine.index) - 1].date()),'%Y-%m-%d') + relativedelta(years=x) for x in range(0, 30)]
        elif(wine.index.freq == "YS"):
            date_list = [datetime.strptime(str(wine.index[len(wine.index) - 1].date()),'%Y-%m-%d') + relativedelta(years=x) for x in range(0, 30)]
        elif(wine.index.freq == "QS"):
            date_list = [datetime.strptime(str(wine.index[len(wine.index) - 1].date()),'%Y-%m-%d') + relativedelta(month=x) for x in [0,3
                ,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51
                ,54,57,60,63,66,69,72,75,78,81,84,87,90]]
        elif(wine.index.freq == "Q"):
            date_list = [wine.index[length_mass - 1].date()
                         + relativedelta(months=x) for x in [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51
                ,54,57,60,63,66,69,72,75,78,81,84,87,90]]

    except:
        date_list = range(length_mass, length_mass + 31)
    future = pd.DataFrame(index=date_list, columns=wine2.columns)
    wine2 = pd.concat([wine2, future])
    print(wine2)
    wine2['forecast'] = best_model.predict(start=length_mass-4, end=length_mass+28)
    plt.figure(figsize=(15, 7))
    wine2.val.plot()
    wine2.forecast.plot(color='r')
    wine3 = wine2[['forecast']]
    wine3.to_excel('Prognoz.xlsx')
    plt.ylabel('Values')
    pylab.savefig('predict.png')
    scale_image('predict.png', 'predict1.png', window_size_X-200)
    img = ImageTk.PhotoImage(Image.open("predict1.png"))
    l_primer.config(image=img)
    l_primer.photo_ref = img
    l_primer.grid()







# Проверка остатков модели

def check_ost():
    clear_root()
    plt.figure(figsize=(17, 9))
    plt.subplot(211)
    best_model.resid[(count_of_diff+count_of_season_diff*period):].plot()

    ax = plt.subplot(212)
    sm.graphics.tsa.plot_acf(best_model.resid[(count_of_diff+count_of_season_diff*period):].values.squeeze(), lags=25, ax=ax)
    print(count_of_diff+count_of_season_diff*period)
    pylab.savefig("ostatki.png")
    bc = stats.ttest_1samp(best_model.resid[(count_of_diff+count_of_season_diff*period):], 0)[1]
    bcc = sm.tsa.stattools.adfuller(best_model.resid[(count_of_diff+count_of_season_diff*period):])[1]
    scale_image('ostatki.png', 'ostatki1.png', window_size_X-200)
    img = ImageTk.PhotoImage(Image.open("ostatki1.png"))
    l_primer.config(image=img)
    l_primer.photo_ref = img
    l_primer.grid()
    l11.grid()
    print("Критерий Стьюдента: p=%f" % bc)
    print("Критерий Дики-Фуллера: p=%f" % bcc)
    global res
    if (bc>0.05 and bcc<0.01):
        ab["text"] = "Если на графике автокорреляции нет лагов сильно отличных от 0, то модель адекватная"
        res = 1
    else:
        ab["text"] = "Модель не является доработанной, попробуйте изменить параметры"
        res = 0
    ab.grid()
    l13.grid()
    btn_progn.grid()


# Введено неверное значение периодичности значений, попробуйте еще раз


def wrong():
    clear_root()
    l10.grid()
    l11.grid()
    aa.grid()
    l13.grid()
    btn_exit.grid()



# ВЫход


def exit_p():
    raise SystemExit




#Построение модели
def postr_models():
    clear_root()
    obrabotka.grid()
    l11.grid()
    btn10.grid()
    

def postr_modelss():
    clear_root()

    global p
    p = int(p_choice.get())
    global P
    P = int(P_choice.get())
    global q
    global Q
    global count_of_diff
    global count_of_season_diff
    global period

    if p>4 and p<7:
        p = [0,1,2,3,4,5,6,7]
    elif p>7:
        p = [0, 1, 2, 3, 4, 5, p - 2, p - 1, p, p + 1]
    elif p<=4:
        p = [0, 1, 2, 3, 4, 5]

    if q>4 and q<7:
        q = [0,1,2,3,4,5,6,7]
    elif q>7:
        q = [0, 1, 2, 3, 4, 5, q - 2, q - 1, q, q + 1]
    elif q<=4:
        q = [0, 1, 2, 3, 4, 5]
    P=[P-1, P, P+1]
    Q=[Q-1, Q, Q+1]
    d = [count_of_diff]
    D = [count_of_season_diff]
    results = []
    global best_aic
    best_aic = float("inf")
    parameters = product(p, q, P, Q, d, D)
    parameters_list = list(parameters)
    length_del = len(parameters_list)
    a=0
    obrabotka2.grid()
    warnings.filterwarnings('ignore')
    global best_model
    global best_param
    for param in parameters_list:
        # try except нужен, потому что на некоторых наборах параметров модель не обучается
        a+=1
        print(a,length_del)
        obrabotka2["text"] = "Построено " + str(a)+" моделей из "+str(length_del)
        try:
            model = sm.tsa.statespace.SARIMAX(wine.val, order=(param[0], param[4], param[1]),
                                              seasonal_order=(param[2], param[5], param[3], period),
                                              enforce_invertibility=False, enforce_stationarity=False).fit(disp=-1)
            print(param, model.aic)
        except:
            print("wrong parameters",a)
            continue
        aic = model.aic
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
    clear_root()
    warnings.filterwarnings('default')
    wine['model'] = best_model.fittedvalues[(count_of_diff+count_of_season_diff*period):]
    params = best_param
    models_param["text"] = "Лучшей моделью среди построенных оказалась модель со следующими параметрами:\n" \
                           "p = "+str(params[0])+ " d = "+str(params[4])+" q = "+str(params[1])+" P = "+str(params[2])+" D = "+str(params[5])+ " Q = "+str(params[3])
    models_param.grid()
    l12.grid()
    l13.grid()
    plt.figure(figsize=(15, 7))
    wine.val.plot()
    wine.model.plot(color='r')
    plt.ylabel('Values')
    pylab.savefig("graf4.png")
    scale_image('graf4.png', 'graf4_1.png', window_size_X - 200)
    img = ImageTk.PhotoImage(Image.open("graf4_1.png"))
    p_c_autocor.config(image=img)
    p_c_autocor.photo_ref = img
    p_c_autocor.grid()
    l11.grid()
    btn_cont.grid()






# Инструкция по выбору параметров




def istr_v_p():
    clear_root()
    instructionn.grid()
    l12.grid()
    btn7.grid()
    l13.grid()
    scale_image('primer_autocor.png', 'primer_autocor1.png', window_size_X-200)
    img = ImageTk.PhotoImage(Image.open("primer_autocor1.png"))
    l_primer.config(image=img)
    l_primer.photo_ref = img
    l_primer.grid()
    btn7.grid()
    print(length_mass)

#Выбор параметров q и Q


def vybor_q_and_Q():
    clear_root()
    plt.figure(figsize=(15, 4))
    ax = plt.subplot(211)
    q_ch.grid()
    q_choice.grid()
    l12.grid()
    Q_ch.grid()
    Q_choice.grid()
    l13.grid()
    sm.graphics.tsa.plot_acf(wine.sales_box[(count_of_diff+count_of_season_diff*period):].values.squeeze(), lags=27, ax=ax)
    pylab.savefig('graf3.png')
    scale_image('graf3.png', 'graf3_1.png', window_size_X - 200)
    img = ImageTk.PhotoImage(Image.open("graf3_1.png"))
    q_autocor.config(image=img)
    q_autocor.photo_ref = img
    q_autocor.grid()
    l11.grid()
    btn8.grid()




#Выбор параметров p and P


def vybor_p_and_P():
    global q
    global Q
    q = int(q_choice.get())
    Q = int(Q_choice.get())
    clear_root()
    plt.figure(figsize=(15, 4))
    ax = plt.subplot(212)
    p_ch.grid()
    p_choice.grid()
    l12.grid()
    P_ch.grid()
    P_choice.grid()
    l13.grid()
    sm.graphics.tsa.plot_pacf(wine.sales_box[(count_of_diff+count_of_season_diff*period):].values.squeeze(), lags=27, ax=ax)
    pylab.savefig('graf3.png')
    scale_image('graf3.png', 'graf3_1.png', window_size_X - 200)
    img = ImageTk.PhotoImage(Image.open("graf3_1.png"))
    p_c_autocor.config(image=img)
    p_c_autocor.photo_ref = img
    p_c_autocor.grid()
    l11.grid()
    btn9.grid()

# Очистка root'a

def clear_root():
    list = root.grid_slaves()
    for l in list:
        l.grid_forget()


# Первичное преобразование ряда


def preobraz_ryada():
    global period
    global count_of_diff
    count_of_diff= 1
    global count_of_season_diff
    count_of_season_diff=0
    clear_root()
    wine['sales_box']= wine.val - wine.val.shift(1)
    wine2 = wine['sales_box'][1:]
    m = (len(wine.sales_box)) // 2
    r1 = sm.stats.DescrStatsW(wine2[m:])
    r2 = sm.stats.DescrStatsW(wine2[1:m])
    d=sm.stats.CompareMeans(r1, r2).ttest_ind()
    if (sm.tsa.stattools.adfuller(wine.sales_box[1:])[1] > 0.001 or d[1]<=0.1):
        wine['sales_box']= wine.sales_box - wine.sales_box.shift(period)
        print(wine)
        count_of_season_diff+=1
        plt.figure(figsize=(15, 7))
        wine.sales_box[(1+period):].plot()
        plt.ylabel('value')
        pylab.savefig('graf2.png')
        scale_image('graf2.png', 'graf2_1.png', window_size_X)
        img = ImageTk.PhotoImage(Image.open("graf2_1.png"))
        l3.config(image=img)
        l3.photo_ref = img
        b=sm.tsa.stattools.adfuller(wine.sales_box[(1+period*count_of_season_diff):])[1]
        global c
        d=c+str(b)
        kr_dik_fuler["text"]=d
        l3.grid_forget()
        kr_dik_fuler.grid()
        if (b>0.01):
            stationary_ts["text"] = "Ряд не стационарен, необходимы преобразования"
            stationary_ts.grid()
            l9.grid()
            btn6["command"]=differents()
            btn6.grid()
        elif(b<=0.01 and b>=0):
            stationary_ts["text"] = "Ряд стационарен, возможно прогнозирование"
            stationary_ts.grid()
            l10.grid()
            l3.grid()
            l12.grid()
            btn5.grid()
    else:
        plt.figure(figsize=(15, 7))
        wine.sales_box[(1):].plot()
        plt.ylabel('value')
        pylab.savefig('graf2.png')
        scale_image('graf2.png', 'graf2_1.png', window_size_X)
        img = ImageTk.PhotoImage(Image.open("graf2_1.png"))
        l3.config(image=img)
        l3.photo_ref = img
        stationary_ts["text"] = "Ряд стационарен, возможно прогнозирование"
        stationary_ts.grid()
        l10.grid()
        l3.grid()
        l12.grid()
        btn5.grid()


#  Функция, меняющая размер изображения под размер экрана


def scale_image(input_image_path,
                output_image_path,
                width=None,
                height=None
                ):
    original_image = pl.Image.open(input_image_path)
    w, h = original_image.size

    if width and height:
        max_size = (width, height)
    elif width:
        max_size = (width, h)
    elif height:
        max_size = (w, height)
    else:
        # No width or height specified
        raise RuntimeError('Width or height required!')

    original_image.thumbnail(max_size, pl.Image.ANTIALIAS)
    original_image.save(output_image_path)


# Функция после кнопки продолжить на 1 экране


def perehod_2_ekran():
    clear_root()
    l10.grid()
    btn3.grid()

#  Отображение графика и проверка на стационарность

def graf_stacion():
    clear_root()
    scale_image('graf1.png', 'graf1_1.png', window_size_X)
    img = ImageTk.PhotoImage(Image.open("graf1_1.png"))
    l3.config(image = img)
    l3.photo_ref = img
    l3.grid()
    a = sm.tsa.stattools.adfuller(wine.val)[1]
    global c
    b = c + str(a)
    kr_dik_fuler["text"] = b
    kr_dik_fuler.grid()
    if (a>0.01):
        stationary_ts["text"] = "Ряд не стационарен, необходимы преобразования"
        stationary_ts.grid()
        l9.grid()
        btn4.grid()
    elif(a<=0.01 and a>=0):
        stationary_ts["text"] = "Ряд стационарен, возможно прогнозирование"
        stationary_ts.grid()
        btn5.grid()
    else:
        stationary_ts["text"] = "Некооректное значение"
        stationary_ts.grid()






#  Обратное преобразование Бокса-Кокса




def invboxcox(y, lmbda):
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda*y+1)/lmbda))


# Входной экран без дат


def welcome2():
    clear_root()
    l00.grid()
    l5.grid()
    l_rasdel.grid()
    rasdel.grid()
    l12.grid()
    l_period.grid()
    l13.grid()
    period_entry.grid()
    l8.grid()
    btn_click.grid()


# Входной экран c датами


def welcome():
    clear_root()
    l0.grid()
    l5.grid()
    l_rasdel.grid()
    rasdel.grid()
    l12.grid()
    l_period.grid()
    l13.grid()
    period_e.grid()
    l8.grid()
    btn.grid()



# Кнопка выбора csv-файла


def click_button2():
    global file_name
    file_name = fd.askopenfilename()
    global wine
    global period
    period = int(period_entry.get())
    wine = pd.read_csv(file_name)
    for i in range(len(wine.index)):
        wine['val'][i] = float(wine['val'][i])
    global length_mass
    print(wine)
    length_mass = len(wine.index)
    print(length_mass)
    plt.figure(figsize=(15, 7))
    wine.val.plot()
    plt.ylabel('srmes')
    pylab.savefig('graf1.png')
    l4.grid()
    l7.grid()
    l2["text"] = wine[:13]
    l2.grid()
    l6.grid()
    btn2.grid()
    l00.grid_forget()
    l5.grid_forget()
    l8.grid_forget()
    l12.grid_forget()
    l13.grid_forget()
    period_entry.grid_forget()
    l_period.grid_forget()
    l_rasdel.grid_forget()
    rasdel.grid_forget()
    btn_click.grid_forget()


# Кнопка выбора csv-файла



def click_button():
    global file_name
    file_name = fd.askopenfilename()
    global wine
    global period
    z=rasdel.get()
    if (period_e.get() == "Раз в день"):

        OPTIONS = ["Раз в день", "Раз в неделю", "Раз в начало месяца",
                   "Раз в конец месяца", "Раз в начало года", "Раз в конец года",
                   "Раз в начало квартала", "Раз в конец квартала"]

    wine = pd.read_csv(file_name,z, index_col=[0], parse_dates=[0], dayfirst=True)
    for i in range(len(wine.index)):
        wine['val'][i] = float(wine['val'][i])
    global length_mass
    print(wine)
    length_mass = len(wine.index)
    print(length_mass)
    plt.figure(figsize=(15, 7))
    aaaa = 0
    if (period_e.get() == "Раз в день"):
        wine.index.freq = "D"
        period = 7
    elif (period_e.get()=="Раз в неделю"):
        wine.index.freq = "W"
        period = 52
    elif (period_e.get()=="Раз в начало месяца"):
        wine.index.freq = "MS"
        period = 12
    elif (period_e.get() == "Раз в конец месяца"):
        wine.index.freq = "M"
        period = 12
    elif (period_e.get() == "Раз в конец года"):
        wine.index.freq = "Y"
        period = 2
    elif (period_e.get() == "Раз в начало года"):
        wine.index.freq = "YS"
        period = 2
    elif (period_e.get() =="Раз в начало квартала"):
        wine.index.freq = "QS"
        period = 4
    elif (period_e.get() == "Раз в конец квартала"):
        wine.index.freq = "Q"
        period = 4
    else:
        aaaa=1
    if aaaa == 0:
        print(wine)
        wine.val.plot()
        plt.ylabel('srmes')
        pylab.savefig('graf1.png')
        l4.grid()
        l7.grid()
        l2["text"] = wine[:13]
        l2.grid()
        l6.grid()
        btn2.grid()
        l0.grid_forget()
        l5.grid_forget()
        l8.grid_forget()
        l12.grid_forget()
        l13.grid_forget()
        period_e.grid_forget()
        l_period.grid_forget()
        l_rasdel.grid_forget()
        rasdel.grid_forget()
        btn.grid_forget()
    else:
        wrong()



#  Создание гланого окна



root = Tk()
root.title("Прогнозирование с помощью модели ARIMA")
screen_size_X=str(root.winfo_screenwidth())
screen_size_Y=str(root.winfo_screenheight())
root.minsize(800,600)
#root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.geometry(screen_size_X+"x"+screen_size_Y)
root.config(bg="white")



#  Метки


l_welcome = Label(text = 'Добро пожаловать!\n\n'
                         'Для работы с программой необходимо иметь файл в формате csv, содержащий временной ряд\n'
                         'В файле должно быть два столбца, 1-й - это дата, второй - это значения\n'
                         'Если временной ряд не содержит дат, тогда в файле должен быть 1 столбец со значениями\n\n\n'
                         'Внимание!!!!!\n'
                         'Для использования файла с датами, необходимо чтобы даты не должны быть пропущены\n'
                         'Например, если данные собираются раз в день, то все дни от 1-го до последнего должны быть в файле\n'
                         'В случае, если это не так, необходимо оставить в файле только столбец со значениями\n'
                         'Для использования файла с датами нажмите кнопку "Файл с датами"\n'
                         'Для использования файла без дат нажмите кнопку "Файл без дат"', background = 'white', font = '16')
l4 = Label(text = 'Файл успешно обработан\nЧасть получившегося ряда представлена ниже', background = 'white', font = '14')
l2 = Label(text = 'Файл выбран, для построения модели ARIMA нажмите "продолжить"', background = 'white')
l1 = Label(text = 'Нажмите кнопку, чтобы выбрать файл с данными для модели')
l00 = Label(text="Добро пожаловать!\n\nДля прогнозирования необходимо иметь файл с историческими данными в формате .csv\n"
                "Файл должен содержать 2 столбца, первым должен идти столбец с датой, время пока не поддерживается\n"
                "Вторым столбцом должен быть столбец со сответствующими значениями\n\n"
                "Сначала введите разделитель, используемый в csv-файле, обычно это ',' или ';'\n\n"
                "Затем, введите предполагаемую периодичность исторических данных\n"
                "\nНапример, если в исторических данных данные раз в месяц, то периодичность будет 12"
                "\nЗатем, для выбора файла нажмите кнопку 'Click me'", background='white', font = '16')
l0 = Label(text="Добро пожаловать!\n\nДля прогнозирования необходимо иметь файл с историческими данными в формате .csv\n"
                "Файл должен содержать 2 столбца, первым должен идти столбец с датой, время пока не поддерживается\n"
                "Вторым столбцом должен быть столбец со сответствующими значениями\n\n"
                "Сначала введите разделитель, используемый в csv-файле, обычно это ',' или ';'\n\n"
                "Затем, выберите из списка периодичность исторических данных (раз в день, неделю и т. д.)\n"
                "На данный момент доступны следующие варианты:\n"
                "Раз в неделю, Раз в месяц (в конце месяца и в начале месяца)\n"
                "Раз в год (в начале года и в конце года), раз в день "
                "(без указания времени)\n"
                "Раз в квартал (в начале или в конце квартала)\n"
                "\nНапример, если в исторических данных данные раз в месяц, то цикл будет 12"
                "\nЗатем, для выбора файла нажмите кнопку 'Click me'", background='white', font = '16')
l5 = Label(text = "Эта метка нужна для отступа, не обращайте внимания",font="16", background = 'white', foreground = 'white')
l15 = Label(text = "Эта метка нужна для отступа, не обращайте внимания",font="16", background = 'white', foreground = 'white')
l14 = Label(text = "Эта метка нужна для отступа, не обращайте внимания",font="16", background = 'white', foreground = 'white')
l13 = Label(text = "Эта метка нужна для отступа, не обращайте внимания",font="16", background = 'white', foreground = 'white')
l12 = Label(text = "Эта метка нужна для отступа, не обращайте внимания",font="16", background = 'white', foreground = 'white')
l11 = Label(text = "Эта метка нужна для отступа, не обращайте внимания",font="16", background = 'white', foreground = 'white')
l10 = Label(text = "Эта метка нужна для отступа, не обращайте внимания",font="16", background = 'white', foreground = 'white')
l9 = Label(text = "Эта метка нужна для отступа, не обращайте внимания",font="16", background = 'white', foreground = 'white')
l8 = Label(text = "Эта метка нужна для отступа, не обращайте внимания",font="16", background = 'white', foreground = 'white')
l7 = Label(text = "Эта метка нужна для отступа, не обращайте внимания",font="16", background = 'white', foreground = 'white')
l6 = Label(text = "Эта метка нужна для отступа, не обращайте внимания",font="16", background = 'white', foreground = 'white')
l_rasdel = Label(text = "Введите разделитель без иных знаков:",background = 'white')
l_period = Label(text= "Введите период без иных знаков:", background = "white")
kr_dik_fuler = Label( font =14, background = 'white')
stationary_ts = Label(font = 14, background = 'white')
l3 = Label(background = 'white')
instructionn = Label(text = "Далее осуществляются выбор параметров q и Q\n\nПараметр q равен номеру последнего лага, сильно отличного от 0 на "
                            "графике автокорреляции\nТо есть, номер последнего лага, выходящего за пределы синего коридора\nПараметр Q равен номеру последнего"
                            " сезонного лага, сильно отличного от 0\n\nСезонный лаг - это лаг, кратный сезонному периоду\nДля периода 12 - это лаги 12, 24 и т. д.\n\n"
                            "После выбора q и Q будет осуществлен выбор параметров p и P\nЭто делается с помощью графика частичной автокорреляционной функции\n"
                            "Выбор происходит по аналогии с параметрами q и Q\np - это номер последнего лага, сильно отличного от 0\n"
                            "В свою очередь, P - это номер последнего сезонного лага, сильно отличного от 0\n"
                            "q и Q выбираются с помощью автокорреляционной функции\n"
                            "p и P выбираются с помощью частичной автокорреляционной функции\n"
                            "Для быстрого прогнозирования введите значения параметров, равные нулю\n\nНиже пример графика автокорреляционной функции",
                     background = "white")
q_ch = Label(text = "Введите значение q",font="16", background = 'white')
Q_ch = Label(text = "Введите значение Q",font="16", background = 'white')
p_ch = Label(text = "Введите значение p",font="16", background = 'white')
P_ch = Label(text = "Введите значение P",font="16", background = 'white')
l_primer = Label()
q_autocor = Label()
Q_autocor = Label()
p_c_autocor = Label()
P_c_autocor = Label()
obrabotka = Label(text = "Внимание!\n"
                         "Процесс построения модели может занять до нескольких часов\n"
                         "Все это время приложение будет не отвечать\n"
                         "После завершения вычислений появится экран с графиком построенной модели", font = '24', background = "white")
obrabotka2 = Label(background = "white")
models_param= Label(background = "white")
aa = Label(text = "Введено неверное значение периодичности, выберите из выпадающего списка верное значение", background = 'white')
ab = Label(background = 'white')


# Кнопки



btn = Button(text="Click Me", background="#363636", foreground="white",
             font="16", command=click_button)
btn_click = Button(text="Click Me", background="#363636", foreground="white",
             font="16", command=click_button2)
btn2 = Button(text="Продолжить", background="#363636", foreground="white",
             padx="20", pady="8", font="16", command=perehod_2_ekran)
btn3 = Button(text="Проверить ряд на стационарность",background="#363636", foreground="white",
             padx="20", pady="8", font="16", command=graf_stacion)
btn4 = Button(text="Преобразовать ряд",background="#363636", foreground="white",
             padx="20", pady="8", font="16", command=preobraz_ryada)
btn5 = Button(text="Перейти к выбору параметров для модели",background="#363636", foreground="white",
             padx="20", pady="8", font="16", command=istr_v_p)
btn6 = Button(text="Продолжить преобразование",background="#363636", foreground="white",
             padx="20", pady="8", font="16", command = differents)
btn7 = Button(text="Перейти к выбору q и Q",background="#363636", foreground="white",
             padx="20", pady="8", font="16", command = vybor_q_and_Q)
btn8 = Button(text="Перейти к выбору p и P",background="#363636", foreground="white",
             padx="20", pady="8", font="16", command = vybor_p_and_P)
btn9 = Button(text="Перейти к построению моделей",background="#363636", foreground="white",
             padx="20", pady="8", font="16", command = postr_models)
btn10 = Button(text="Перейти к построению моделей",background="#363636", foreground="white",
             padx="20", pady="8", font="16", command = postr_modelss)
btn_exit = Button(text="Выход",background="#363636", foreground="white",
             padx="20", pady="8", font="16", command = exit_p)
btn_cont = Button(text="Проверить модель на адекватность",background="#363636", foreground="white",
             padx="20", pady="8", font="16", command = check_ost)
btn_progn = Button(text="Построить прогноз",background="#363636", foreground="white",
             padx="20", pady="8", font="16", command = check_ost)
btn_progn = Button(text="Построить прогноз",background="#363636", foreground="white",
             padx="20", pady="8", font="16", command =prognoz)
btn_with_date = Button(text="Файл с датами",background="#363636", foreground="white",
             padx="20", pady="8", font="16", command =welcome)
btn_without_date = Button(text="Файл без дат",background="#363636", foreground="white",
             padx="20", pady="8", font="16", command =welcome2)



# Поле для ввода



rasdel = Entry(width=10)
q_choice = Entry(width=10)
Q_choice = Entry(width=10)
p_choice = Entry(width=10)
P_choice = Entry(width=10)
period_entry = Entry(width=10)


# Выпадающие списки


OPTIONS = ["Раз в день", "Раз в неделю", "Раз в начало месяца",
           "Раз в конец месяца", "Раз в начало года", "Раз в конец года",
           "Раз в начало квартала", "Раз в конец квартала"]
period_e = ttk.Combobox(root, values = OPTIONS)



#   Начальный экран

l_welcome.grid()
l11.grid()
l12.grid()
btn_with_date.grid()
l10.grid()
btn_without_date.grid()




length_mass = 0
count_of_diff = 0
count_of_season_diff=0
period = 0

q=0
Q=0
p=0
P=0

root.update_idletasks()

window_size_X=root.winfo_width()
window_size_Y=root.winfo_height()

root.mainloop()











