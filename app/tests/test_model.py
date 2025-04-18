import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from app.ml.models.model import BaseModel, ModelFactory
from app.ml.data.preprocessing import DataPreprocessor


# Tạo dữ liệu mẫu cho các tests
@pytest.fixture
def sample_data():
    """Fixture tạo dữ liệu mẫu cho các tests."""
    # Tạo dữ liệu đơn giản cho bài toán phân loại
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 3, 100)  # 3 classes
    return X, y


@pytest.fixture
def preprocessed_data(sample_data):
    """Fixture tạo dữ liệu đã được tiền xử lý."""
    X, y = sample_data

    # Chuyển đổi sang DataFrame
    X_df = pd.DataFrame(X, columns=[f'feature{i}' for i in range(5)])
    y_series = pd.Series(y, name='target')

    # Tiền xử lý
    preprocessor = DataPreprocessor()
    X_processed = preprocessor.fit_transform(X_df, y_series)
    y_processed = preprocessor.transform_target(y_series)

    return X_processed.values, y_processed, preprocessor


def test_model_factory_create():
    """Test tạo mô hình bằng ModelFactory."""
    # Kiểm tra tạo Random Forest
    rf_model = ModelFactory.create_model("random_forest")
    assert rf_model is not None
    assert hasattr(rf_model, 'fit')
    assert hasattr(rf_model, 'predict')

    # Kiểm tra tạo SVM
    svm_model = ModelFactory.create_model("svm")
    assert svm_model is not None
    assert hasattr(svm_model, 'fit')
    assert hasattr(svm_model, 'predict')

    # Kiểm tra tạo Neural Network
    nn_model = ModelFactory.create_model("neural_network")
    assert nn_model is not None
    assert hasattr(nn_model, 'fit')
    assert hasattr(nn_model, 'predict')

    # Kiểm tra với loại không xác định
    default_model = ModelFactory.create_model("unknown_type")
    assert default_model is not None
    assert hasattr(default_model, 'fit')
    assert hasattr(default_model, 'predict')


def test_base_model_creation():
    """Test khởi tạo BaseModel."""
    # Khởi tạo với loại mặc định
    model = BaseModel()
    assert model.model_type == "random_forest"
    assert model.model is None

    # Khởi tạo với loại cụ thể
    model = BaseModel(model_type="svm", model_id="test_model")
    assert model.model_type == "svm"
    assert model.model_id == "test_model"
    assert model.model is None


def test_base_model_fit_predict(preprocessed_data):
    """Test huấn luyện và dự đoán với BaseModel."""
    X, y, _ = preprocessed_data

    # Khởi tạo mô hình
    model = BaseModel(model_type="random_forest")

    # Tạo và huấn luyện mô hình
    model.create(n_features=X.shape[1], n_classes=len(np.unique(y)))
    model.fit(X, y)

    # Kiểm tra model đã được tạo và huấn luyện
    assert model.model is not None
    assert hasattr(model.model, 'predict')

    # Thực hiện dự đoán
    y_pred = model.predict(X)
    assert y_pred is not None
    assert len(y_pred) == len(y)

    # Kiểm tra predict_proba
    y_proba = model.predict_proba(X)
    assert y_proba is not None
    assert y_proba.shape[0] == len(y)
    assert y_proba.shape[1] == len(np.unique(y))

    # Đánh giá mô hình
    metrics = model.evaluate(X, y)
    assert metrics is not None
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


def test_base_model_save_load(preprocessed_data):
    """Test lưu và load BaseModel."""
    X, y, _ = preprocessed_data

    # Khởi tạo và huấn luyện mô hình
    model = BaseModel(model_type="random_forest", model_id="test_save_load")
    model.create(n_features=X.shape[1], n_classes=len(np.unique(y)))
    model.fit(X, y)

    # Tạo thư mục tạm để lưu mô hình
    with tempfile.TemporaryDirectory() as tmpdir:
        # Lưu mô hình
        model_path = os.path.join(tmpdir, "test_model.joblib")
        model.save(model_path)

        # Kiểm tra file đã được tạo
        assert os.path.exists(model_path)

        # Load mô hình
        loaded_model = BaseModel.load(path=model_path)

        # Kiểm tra mô hình đã được load
        assert loaded_model is not None
        assert loaded_model.model_type == model.model_type
        assert loaded_model.is_tensorflow == model.is_tensorflow

        # So sánh dự đoán
        y_pred_original = model.predict(X)
        y_pred_loaded = loaded_model.predict(X)
        assert np.array_equal(y_pred_original, y_pred_loaded)