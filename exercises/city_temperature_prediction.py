import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df.dropna().drop_duplicates()
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df['Year'] = df['Year'].astype(str)
    return df[df['Temp'] > -40]

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df_IL = df[df['Country'] == "Israel"]

    px.scatter(df_IL, x='DayOfYear', y='Temp', color='Year',
                         color_discrete_sequence=px.colors.qualitative.Plotly,
                         title="Temperature in Israel as a function of day of year",
                         labels={'x':"Day of year", 'y':"Temp"})\
        .write_image("./city_temperature_plots/DayOfYear_Israel.png")

    px.bar(df_IL.groupby('Month', as_index=False).agg('std'), x='Month', y='Temp',
                   title="Standard deviation of temperature per month in Israel")\
        .write_image("./city_temperature_plots/STD_month_Israel.png")

    # Question 3 - Exploring differences between countries
    line_df = df.groupby(['Month', 'Country'], as_index=False)['Temp'].agg(['mean', 'std'])
    line_df = line_df.reset_index()
    px.line(line_df, x='Month', y='mean', color='Country', error_y='std',
                   title="Average monthly temperature",
            labels={'mean':'Mean temperature'}).write_image("./city_temperature_plots/Average_monthly_temperature.png")

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(df_IL['DayOfYear'], df_IL['Temp'])
    k_range = list(range(1,11))
    loss = np.zeros(len(k_range), dtype=float)
    for k in k_range:
        model = PolynomialFitting(k)
        model.fit(train_X, train_y)
        loss[k-1] = round(model.loss(test_X, test_y),2)
    print(loss)
    px.bar(x=k_range, y=loss, text_auto=True, title="Test error recorded for different degrees of the model",
           labels={'x':'Degree', 'y':'Error'}).write_image("./city_temperature_plots/different_k_values.png")


    # Question 5 - Evaluating fitted model on different countries
    X_IL, y_IL = df_IL['DayOfYear'], df_IL['Temp']
    model_IL = PolynomialFitting(5)
    model_IL.fit(X_IL, y_IL)
    other_countries = df[df['Country'] != 'Israel']['Country'].unique().tolist()
    loss_countries = pd.DataFrame(columns=['Country', 'Loss'])
    for i, c in enumerate(other_countries):
        new_row = {'Country': c, 'Loss': model_IL.loss(df[df['Country'] == c]['DayOfYear'],
                                                      df[df['Country'] == c]['Temp'])}
        loss_countries = loss_countries.append(new_row, ignore_index=True)
    px.bar(loss_countries, x='Country', y='Loss', color='Country',
           title="Model’s error over each of the other countries")\
        .write_image("./city_temperature_plots/different_countries_pred.png")
