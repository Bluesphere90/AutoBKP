import pytest
import numpy as np
import pandas as pd
from app.ml.data.preprocessing import DataPreprocessor
from app.ml.data.validation import DataValidator


# Tạo dữ liệu mẫu cho các tests
@pytest.fixture
def sample_data():
    """Fixture tạo dữ liệu mẫu cho các tests."""
    # Tạo DataFrame với một số missing values và outliers
    data = {
        'feature1': [1, 2, np.nan, 4, 5, 6, 100],  # Có missing value và outlier
        'feature2': [10, 20, 30, 40, 50, 60, 70],
        'feature3': ['A', 'B', 'C', 'A', 'B', None, 'D'],  # Có missing value
        'target': [0, 1, 2, 0, 1, 2, 0]  # Target có 3 classes
    }
    df = pd.DataFrame(data)
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y


def test_preprocessor_initialization():
    """Test khởi tạo DataPreprocessor."""
    preprocessor = DataPreprocessor()
    assert preprocessor is not None
    assert preprocessor.scaling_method == "standard"
    assert preprocessor.encoding_method == "one_hot"
    assert preprocessor.missing_values_method == "mean"


def test_preprocessor_fit_transform(sample_data):
    """Test fit_transform của DataPreprocessor."""
    X, y = sample_data

    # Khởi tạo preprocessor
    preprocessor = DataPreprocessor({
        'scaling': 'standard',
        'categorical_encoding': 'one_hot',
        'missing_values': 'mean'
    })

    # Fit và transform dữ liệu
    X_transformed = preprocessor.fit_transform(X, y)

    # Kiểm tra kết quả
    assert X_transformed is not None
    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[0] == X.shape[0]  # Số hàng giữ nguyên
    assert not X_transformed.isnull().any().any()  # Không còn missing values

    # Kiểm tra one-hot encoding
    assert 'feature3_A' in X_transformed.columns
    assert 'feature3_B' in X_transformed.columns
    assert 'feature3_C' in X_transformed.columns
    assert 'feature3_D' in X_transformed.columns

    # Kiểm tra scaling
    assert -3 < X_transformed['feature1'].min() < 3
    assert -3 < X_transformed['feature1'].max() < 3
    assert -3 < X_transformed['feature2'].min() < 3
    assert -3 < X_transformed['feature2'].max() < 3


def test_preprocessor_transform_target(sample_data):
    """Test transform_target của DataPreprocessor."""
    X, y = sample_data

    # Khởi tạo preprocessor với label encoding
    preprocessor = DataPreprocessor({
        'scaling': 'standard',
        'categorical_encoding': 'label',
        'missing_values': 'mean'
    })

    # Fit và transform dữ liệu
    preprocessor.fit(X, y)
    y_transformed = preprocessor.transform_target(y)

    # Kiểm tra kết quả
    assert y_transformed is not None
    assert isinstance(y_transformed, np.ndarray)
    assert len(y_transformed) == len(y)

    # Kiểm tra inverse transform
    y_inverse = preprocessor.inverse_transform_target(y_transformed)
    assert np.array_equal(y_inverse, y.values)


def test_preprocessor_save_load(sample_data, tmpdir):
    """Test save và load của DataPreprocessor."""
    X, y = sample_data

    # Khởi tạo và fit preprocessor
    preprocessor = DataPreprocessor()
    preprocessor.fit_transform(X, y)

    # Lưu preprocessor
    save_path = tmpdir.join("preprocessor_test.joblib")
    preprocessor.save(str(save_path))

    # Load preprocessor
    loaded_preprocessor = DataPreprocessor.load(str(save_path))

    # Kiểm tra kết quả
    assert loaded_preprocessor is not None
    assert loaded_preprocessor.scaling_method == preprocessor.scaling_method
    assert loaded_preprocessor.encoding_method == preprocessor.encoding_method

    # Kiểm tra transform với cả hai preprocessors
    X_transformed_original = preprocessor.transform(X)
    X_transformed_loaded = loaded_preprocessor.transform(X)

    # So sánh kết quả
    pd.testing.assert_frame_equal(X_transformed_original, X_transformed_loaded)


def test_vietnamese_preprocessing(sample_data):
    """Test xử lý dữ liệu tiếng Việt."""
    X, y = sample_data

    # Thêm một số cột text tiếng Việt vào dữ liệu mẫu
    X['text_col'] = ['Công ty TNHH A', 'Công ty TNHH B', 'Công ty Cổ phần C',
                     'CÔNG TY D', 'Công Ty E', 'Công ty F', 'Công ty G']
    X['mst'] = ['0123456789', '9876543210', '1234567890',
                '0987654321', '1122334455', '5544332211', '1029384756']

    # Khởi tạo preprocessor với vietnamese_processing=True
    config = {
        'scaling': 'standard',
        'categorical_encoding': 'one_hot',
        'missing_values': 'mean',
        'vietnamese_processing': True,
        'tax_id_processing': True
    }
    preprocessor = DataPreprocessor(config)

    # Fit và transform dữ liệu
    X_transformed = preprocessor.fit_transform(X, y)

    # Kiểm tra kết quả
    assert X_transformed is not None
    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[0] == X.shape[0]  # Số hàng giữ nguyên

    # Kiểm tra các cột mới từ xử lý mã số thuế
    assert 'mst_length' in X_transformed.columns
    assert 'mst_is_branch' in X_transformed.columns
    assert 'mst_province_code' in X_transformed.columns

    # Kiểm tra encoding của cột text tiếng Việt
    assert 'text_col_encoded' in X_transformed.columns