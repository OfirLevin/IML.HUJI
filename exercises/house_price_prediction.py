from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    if y is not None:
        null_index = y[y.isna()].index.tolist()
        X.drop(index=null_index, inplace=True, axis=0)
        y = y.dropna()
        negative_index = y[y <= 0].index.tolist()
        X.drop(index=negative_index, inplace=True, axis=0)
        y = y[y > 0]

    X.drop(columns=["id", "long", "lat", "date", "sqft_living15", "sqft_lot15"], inplace=True)
    if y is None:
        for column in ['bedrooms', 'bathrooms']:
            X = X[X[column] > 0]

    X = pd.get_dummies(X, columns=["zipcode"], prefix="zipcode")

    X["decade_built"] = X["yr_built"] // 10 * 10
    X.drop(columns="yr_built", inplace=True)
    X = pd.get_dummies(X, columns=["decade_built"], prefix="decade_built")

    X["decade_renovated"] = X["yr_renovated"] // 10 * 10
    X.drop(columns="yr_renovated", inplace=True)
    X = pd.get_dummies(X, columns=["decade_renovated"], prefix="decade_renovated")
    X.rename(columns={"decade_renovated_0.0" : "not_renovated"})

    if y is not None:
        diff_index = y.index.difference(X.index).tolist()
        y.drop(index=diff_index, inplace=True)
        return X, y
    return X



def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y_std = np.std(y)
    X = X[X.columns.drop(list(X.filter(regex='decade|zipcode', axis=1)))]

    for feature in X.columns:
        feature_std = np.std(X[feature])
        pearson_corr = np.cov(y, X[feature])[0, 1] / (y_std * feature_std)
        fig = px.scatter(x=X[feature], y=y, title=f"Correlation between {feature} and response <br>with Pearson "
                                                  f"Correlation {str(pearson_corr)}",
                         labels=dict(x=f"{feature} values", y="Response values"))
        fig.update_layout(title_x=0.5)

        fig.write_image(output_path + "/" + feature + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    train_X, train_y, test_X, test_y = split_train_test(df.drop(columns="price", inplace=False), df["price"])

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)
    test_X = preprocess_data(test_X)
    test_extra_cols = test_X.columns.difference(train_X.columns).tolist()
    test_X.drop(index=test_extra_cols, inplace=True, axis='columns')
    missing_cols = train_X.columns.difference(test_X.columns).tolist()
    for col in missing_cols:
        test_X[col] = 0
    df1 = test_X.reindex(columns=train_X.columns)


    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y, "./feature_evaluation")

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    p_list = list(range(10, 101))
    mean_loss_all, std_loss_all = np.zeros(len(list(range(10, 101)))), np.zeros(len(list(range(10, 101))))
    for p in p_list:
        loss_arr_p = np.zeros(10)
        for j in range(10):
            sample_X = train_X.sample(frac=p / 100.0)
            sample_y = train_y.filter(items=sample_X.index)
            model = LinearRegression()
            model.fit(sample_X, sample_y)
            loss_arr_p[j] = model.loss(test_X, test_y)
        mean_loss_all[p-10] = np.mean(loss_arr_p)
        std_loss_all[p-10] = np.std(loss_arr_p)
    fig = go.Figure([go.Scatter(x=p_list, y=mean_loss_all, marker=dict(color="darkslateblue")),
                    go.Scatter(x=p_list, y=mean_loss_all + 2*std_loss_all,
                               mode='lines',
                               line=dict(width=0),
                               showlegend=False),
                    go.Scatter(x=p_list, y=mean_loss_all - 2*std_loss_all,
                               line=dict(width=0),
                               mode='lines',
                               fillcolor='rgba(20, 100, 200, 0.3)',
                               fill='tonexty',
                               showlegend=False)],
                    layout=go.Layout(title="Mean loss as a function of training set percentage",
                                     xaxis=dict(title="Percentage of sample from training set"),
                                     yaxis=dict(title="MSE Over Test Set"),
                                     showlegend=False)
                    )
    fig.show()
    fig.write_image("./MSE_percentage.png")
