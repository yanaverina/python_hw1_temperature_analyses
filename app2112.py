import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import datetime as dt


st.set_page_config(page_title='Анализ температур в городах', layout="wide")

def get_weather(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    return response.json()

def build_scatter_plot(temperatures, seasons):
    df = pd.DataFrame({'season': seasons, 'temperature': temperatures})
    fig = px.scatter(df, x='season', y='temperature', title='Распределение температуры по сезонам', labels={'season': 'Сезон', 'temperature': 'Температура (°C)'})
    return fig

def build_timeseries_plot(df):
    fig = px.line(df, x="timestamp", y=["temperature", "rolling_avg_30_days"],labels={"timestamp": "Дата", "value": "Температура (°C)"},
        title="Распределение температуры во времени",template="plotly_white", color_discrete_sequence=['green', 'orange'])
    
    fig.add_scatter(x=df.query("anomaly == 1")["timestamp"], y=df.query("anomaly == 1")["temperature"], 
    mode="markers", marker=dict(color="red", size=10),name="Аномалии")
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["mean_temp"] + 2 * df["std_temp"], line=dict(color="blue"), name="Среднее + 2*σ"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["mean_temp"] - 2 * df["std_temp"], line=dict(color="purple"), name="Среднее - 2*σ"))
    fig.update_layout(legend_title_text="Показатели",legend=dict(orientation="h",yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=50, r=0, t=40, b=30))
    return fig

def calculate_trend(df):
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.toordinal())
    X = df['timestamp'].values.reshape(-1, 1)
    y = df['temperature'].values
    model = LinearRegression()
    model.fit(X, y)
    trend = model.coef_[0]
    
    return trend

def anomaly(temperature, mean_t, std_t):
    left_boundary = mean_t - 2*std_t
    rigrt_boundary = mean_t + 2*std_t
    if temperature > rigrt_boundary or temperature < left_boundary:
        return 1
    else:
        return 0
        
def processing_data(df_main, selected_city):
    df = df_main[df_main['city'] == selected_city]
    df["rolling_avg_30_days"] = df["temperature"].rolling(window=30).mean()
    df['rolling_std'] = df['rolling_avg_30_days'].std()
    df['mean_temp'] = df.groupby(['city','season'])['temperature'].transform('mean')
    df['std_temp'] = df.groupby(['city','season'])['temperature'].transform('std')
    df['anomaly'] = df.apply(lambda x: anomaly(x.temperature, x.mean_temp, x.std_temp), axis=1)

    df_stats = pd.DataFrame()
    df_stats = df.groupby(by=['city', 'season']).agg({'temperature':['mean','std', 'max', 'min']})
    return df_stats, df


uploaded_file = st.file_uploader('Выберите csv', type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write('Данные загружены.')
else:
    st.write('Пожалуйста, загрузите csv.')

if uploaded_file is not None:
    city = data['city'].unique()
    selected_city = st.selectbox("Выберите город:", city)
    st.write(f'Вы выбрали: {selected_city}')

    st.write(f'<p><strong>Описательная статистика температур {selected_city}</strong></p>', unsafe_allow_html=True)
    stats, data = processing_data(data, selected_city)
    st.write(stats)

    st.write(f'<p><strong>Тренд температур для {selected_city}</strong></p>', unsafe_allow_html=True)
    trend = calculate_trend(data[data['city'] == selected_city])
    if trend > 0:
        st.write(f'Позитивный тренд для города {selected_city}')
    elif trend < 0:
        st.write(f'Негативный тренд для города {selected_city}')

    api_key = st.text_input('Введите ваш API-ключ OpenWeatherMap:')

    if api_key:
        weather_data = get_weather(api_key, selected_city)
                
        if weather_data.get('cod') == 200:
            temp = weather_data['main']['temp']
            st.write(f'<p><strong>Аномальность текущей температуры {selected_city}</strong></p>', unsafe_allow_html=True)  
            st.write(f"Текущая температура в {selected_city}: {temp}°C")

            stats = stats.reset_index()
            anomaly_result = anomaly(temp, stats[stats['season'] == 'winter']['temperature']['mean'].iloc[0], 
            stats[stats['season'] == 'winter']['temperature']['std'].iloc[0])
            if anomaly_result == 1:
                st.write(f'Температура {temp} является аномальной для {selected_city}.')
            else:
                st.write(f'Температура {temp} не является аномальной для {selected_city}.')

        elif weather_data.get('cod') == 401:
            st.error(weather_data['message'])
        else:
            st.error(f"Произошла ошибка: {weather_data.get('message', 'Неизвестная ошибка')}")
    else:
        st.warning('Пожалуйста, введите API-ключ.')

    st.plotly_chart(build_scatter_plot(data[data['city'] == selected_city]['temperature'], data[data['city'] == selected_city]['season']))
    st.plotly_chart(build_timeseries_plot(data[data['city'] == selected_city]))