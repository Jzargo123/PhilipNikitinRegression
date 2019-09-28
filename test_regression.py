import pytest
import numpy as np

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import data_frames, series,  columns
from hypothesis.strategies import composite
from Regression.regression import LinearRegression
from sklearn.metrics import r2_score



def test_incorrect_path():
    model = LinearRegression(reg_l1=0.0, reg_l2=0.0, optimizator='L-BFGS-B', intercept=False)
    with pytest.raises(Exception):
        model.load_model('../tmp/incorrect.path')


@composite
def data_generator(draw):
    df = draw(data_frames(columns=columns(names_or_number=[str(i) for i in range(10)], dtype=float,
                                elements=st.floats(allow_infinity=False, max_value=1e+30))))
    reg_l1 = draw(st.floats())
    reg_l2 = draw(st.floats())
    optimizator = draw(st.sampled_from(['L-BFGS-B', 'BFGS']))
    intercept = draw(st.booleans())
    return df, reg_l1, reg_l2, optimizator, intercept


@given(
 data=data_generator()
)
def test_hyperparams(data):
    df = data[0]
    reg_l1 = data[1]
    reg_l2 = data[2]
    optimizator = data[3]
    intercept = data[4]
    y = df[df.columns[-1]]
    X = df[df.columns[:-1]]
    model = LinearRegression(reg_l1=reg_l1, reg_l2=reg_l2, optimizator=optimizator, intercept=intercept)
    if y.shape[0] == 0:
        with pytest.raises(Exception):
            model.fit(X, y)
    else:
        model.fit(X.values, y.values)
        model.predict(X.values)
        model.save_model('../tmp/regression.model')
        model.load_model('../tmp/regression.model')
        assert True


@composite
def generate_model(draw):
    X = draw(arrays(shape=(11, 10), elements=st.floats(allow_infinity=False, min_value=-1e3,
                                                       max_value=1e3), dtype=np.float))
    w = draw(arrays(shape=(10), elements=st.floats(allow_infinity=False, max_value=1e3,
                                                   min_value=-1e3), dtype=np.float))
    return X, w

@given(
    data=generate_model()
)
def test_model(data):
    X, w = data
    y = X.dot(w)
    model = LinearRegression(reg_l1=0.0, reg_l2=0.0, optimizator='L-BFGS-B', intercept=False)
    model.fit(X, y)
    y_predict = model.predict(X)
    assert r2_score(y, y_predict) > 0.8 or np.sum(np.abs(y - y_predict)) < 1e-3



@composite
def generate_model_with_intercept(draw):
    X = draw(arrays(shape=(11, 10), elements=st.floats(allow_infinity=False, min_value=-1e3,
                                                       max_value=1e3), dtype=np.float))
    w = draw(arrays(shape=(10), elements=st.floats(allow_infinity=False, max_value=1e3,
                                                   min_value=-1e3), dtype=np.float))
    const = draw(st.floats(allow_infinity=False, max_value=1e3, min_value=-1e3))
    return X, w, const


@given(
    data=generate_model_with_intercept()
)
def test_model_with_intercept(data):
    X, w, const = data
    y = X.dot(w) + const
    model = LinearRegression(reg_l1=0.0, reg_l2=0.0, optimizator='L-BFGS-B', intercept=True)
    model.fit(X, y)
    y_predict = model.predict(X)
    assert r2_score(y, y_predict) > 0.8 or np.sum(np.abs(y - y_predict)) < 1e-3

