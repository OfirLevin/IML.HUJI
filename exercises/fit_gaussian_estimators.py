from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.express as px
import pandas as pd
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma, N = 10, 1, 1000
    uni_x = np.random.normal(mu, sigma, N)

    uni = UnivariateGaussian()
    uni.fit(uni_x)
    print(round(uni.mu_, 3), round(uni.var_, 3))

    # Question 2 - Empirically showing sample mean is consistent
    dist_from_expectation = []
    samples_size = []
    for i in range(10, 1001, 10):
        tmp_uni = UnivariateGaussian()
        tmp_uni.fit(uni_x[:i])
        dist_from_expectation.append(np.abs(tmp_uni.mu_ - mu))
        samples_size.append(i)

    samples_models_fig = px.scatter(x=samples_size, y=dist_from_expectation,
                                    labels={
                                        "x": "Number of Samples",
                                        "y": "Distance between true value of expectation"},
                                    title="Distance between estimated and true value of expectation as a function of "
                                          "number of samples")
    samples_models_fig.update_traces(marker=dict(color="LightSeaGreen"))
    samples_models_fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    df = pd.DataFrame(np.array([uni_x, uni.pdf(uni_x)])).transpose().sort_values(by=1)
    samples_pdf_fig = px.scatter(df, x=0, y=1,
                                 title="Empirical PDF function under the fitted model")
    samples_pdf_fig.update_traces(marker=dict(color="LightSeaGreen"))
    samples_pdf_fig.update_layout(overwrite=True, xaxis_title="Sample value", yaxis_title="PDF")
    samples_pdf_fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multi_mu = np.array([0, 0, 4, 0])
    multi_cov = np.matrix([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    multi_N = 1000
    multi_x = np.random.multivariate_normal(multi_mu, multi_cov, multi_N)

    multi = MultivariateGaussian()
    multi.fit(multi_x)
    print(np.round(multi.mu_, 3))
    print(np.round(multi.cov_, 3))

    # Question 5 - Likelihood evaluation
    values = np.linspace(-10, 10, 200)
    df = pd.DataFrame(columns=values, index=values)
    for i, f1 in enumerate(df.index):
        for j, f3 in enumerate(df.columns):
            tmp_mu = np.array([f1, 0, f3, 0])
            tmp_log_likelihood = multi.log_likelihood(tmp_mu, multi_cov, multi_x)
            df.iloc[i, j] = tmp_log_likelihood
    log_likelihood_fig = px.imshow(df, title="Log likelihood values over MU of multivariate gaussian",
                                   labels=dict(x="f3 value", y="f1 value", color="Log likelihood"),
                                   color_continuous_scale="plasma")
    log_likelihood_fig.show()

    # Question 6 - Maximum likelihood
    print(np.round(df.stack().index[np.argmax(df.values)], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
