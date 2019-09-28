from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.pandas import data_frames, columns
from Regression.preprocessing import read_data


@given(data_frames(
    columns=columns(names_or_number=[str(i) for i in range(10)], dtype=float,
                    elements=st.floats(allow_infinity=False, max_value=1e+307))
))
def test_get_predict_data(df):
    df.to_csv('../tmp/predict_df.csv')
    res = read_data('../tmp/predict_df.csv', fitting=False)
    assert res.shape == df.shape


@given(data_frames(
    columns=columns(names_or_number=[str(i) for i in range(10)], dtype=float,
                    elements=st.floats(allow_infinity=False, max_value=1e+307))
))
def test_get_train_data(df):
    df.to_csv('../tmp/train_df.csv')
    X, y = read_data('../tmp/train_df.csv', fitting=True, split=False)
    assert X.shape[1] == df.shape[1] - 1
    assert len(y.shape) == 1
    assert y.shape[0] == df.shape[0]


@given(data_frames(
    columns=columns(names_or_number=[str(i) for i in range(10)], dtype=float,
                    elements=st.floats(allow_infinity=False, max_value=1e+307))
))
def test_get_train_splited_data(df):
    df.to_csv('../tmp/train_df.csv')
    if df.shape[0] > 2:
        X_train, X_test, y_train, y_test = read_data('../tmp/train_df.csv', fitting=True, split=True)
        assert X_train.shape[0] + X_test.shape[0] == df.shape[0]
        assert X_train.shape[1] + 1 == df.shape[1]
    else:
        assert 1
