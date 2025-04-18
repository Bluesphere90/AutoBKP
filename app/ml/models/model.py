import os
import joblib
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from app.core.config import settings, get_model_config
from app.core.logging import Logger

logger = Logger(__name__)


def get_available_models() -> List[str]:
    """
    Lấy danh sách các mô hình có sẵn.

    Returns:
        List[str]: Danh sách tên các mô hình
    """
    model_path = settings.MODEL_PATH
    if not os.path.exists(model_path):
        return []

    # Lấy tất cả các file .joblib và .h5 trong thư mục
    model_files = []
    for file in os.listdir(model_path):
        if file.endswith(".joblib") or file.endswith(".h5"):
            model_name = os.path.splitext(file)[0]
            if model_name != "preprocessor":  # Loại bỏ preprocessor
                model_files.append(model_name)

    return model_files


class ModelFactory:
    """
    Factory class để tạo các mô hình phân loại đa lớp.
    """

    @staticmethod
    def create_model(model_type: str, params: Dict[str, Any] = None) -> Any:
        """
        Tạo mô hình dựa trên loại và tham số.

        Args:
            model_type: Loại mô hình ("random_forest", "svm", "neural_network", etc.)
            params: Tham số cho mô hình

        Returns:
            models: Mô hình được khởi tạo
        """
        if params is None:
            # Lấy tham số từ config
            model_config = get_model_config()
            model_params = model_config.get("models", {}).get("params", {})
            params = model_params.get(model_type, {})

        logger.info(f"Creating {model_type} models with parameters: {params}")

        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                min_samples_leaf=params.get("min_samples_leaf", 1),
                random_state=42
            )

        elif model_type == "svm":
            return SVC(
                C=params.get("C", 1.0),
                kernel=params.get("kernel", "rbf"),
                gamma=params.get("gamma", "scale"),
                probability=True,
                random_state=42
            )

        elif model_type == "neural_network":
            if "tensorflow" in params.get("framework", "sklearn"):
                # Tạo mô hình Keras
                return ModelFactory._create_keras_model(params)
            else:
                # Tạo mô hình MLPClassifier của sklearn
                return MLPClassifier(
                    hidden_layer_sizes=params.get("hidden_layers", (100,)),
                    activation=params.get("activation", "relu"),
                    learning_rate_init=params.get("learning_rate", 0.001),
                    max_iter=params.get("epochs", 200),
                    batch_size=params.get("batch_size", 32),
                    random_state=42
                )

        else:
            logger.warning(f"Unknown models type: {model_type}. Using RandomForestClassifier.")
            return RandomForestClassifier(random_state=42)

    @staticmethod
    def _create_keras_model(params: Dict[str, Any]) -> Sequential:
        """
        Tạo mô hình neural network với Keras.

        Args:
            params: Tham số cho mô hình

        Returns:
            models: Mô hình Keras được khởi tạo
        """
        # Lấy tham số
        hidden_layers = params.get("hidden_layers", [128, 64])
        activation = params.get("activation", "relu")
        learning_rate = params.get("learning_rate", 0.001)
        n_classes = params.get("n_classes", 3)  # Mặc định 3 classes
        input_dim = params.get("input_dim", 10)  # Mặc định 10 features
        dropout_rate = params.get("dropout_rate", 0.2)

        # Tạo mô hình tuần tự
        model = Sequential()

        # Input layer
        model.add(Dense(hidden_layers[0], activation=activation, input_dim=input_dim))
        model.add(Dropout(dropout_rate))

        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation=activation))
            model.add(Dropout(dropout_rate))

        # Output layer
        model.add(Dense(n_classes, activation='softmax'))

        # Compile mô hình
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model


class BaseModel:
    """
    Lớp cơ sở cho các mô hình phân loại đa lớp.
    """

    def __init__(
            self,
            model_type: str = "random_forest",
            params: Dict[str, Any] = None,
            model_id: str = None
    ):
        """
        Khởi tạo models.

        Args:
            model_type: Loại mô hình
            params: Các tham số cho mô hình
            model_id: ID của mô hình (để lưu/load)
        """
        self.model_type = model_type
        self.params = params
        self.model_id = model_id or f"{model_type}_model"
        self.model = None
        self.classes_ = None
        self.is_tensorflow = False  # Flag để biết có phải là mô hình TensorFlow không

        logger.info(f"Initialized BaseModel with type: {model_type}, id: {self.model_id}")

    def create(self, n_features: int = None, n_classes: int = None) -> 'BaseModel':
        """
        Tạo mô hình với kiến trúc phù hợp.

        Args:
            n_features: Số lượng features đầu vào
            n_classes: Số lượng classes

        Returns:
            self: instance của models
        """
        # Cập nhật params với số lượng features và classes nếu cần
        if n_features is not None or n_classes is not None:
            if self.params is None:
                self.params = {}

            if n_features is not None and self.model_type == "neural_network":
                self.params["input_dim"] = n_features

            if n_classes is not None and self.model_type == "neural_network":
                self.params["n_classes"] = n_classes

        # Tạo mô hình
        self.model = ModelFactory.create_model(self.model_type, self.params)

        # Kiểm tra xem có phải mô hình TensorFlow không
        self.is_tensorflow = isinstance(self.model, tf.keras.Model)

        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Huấn luyện mô hình trên dữ liệu.

        Args:
            X: Features
            y: Target labels

        Returns:
            self: instance đã được huấn luyện
        """
        logger.info(f"Training {self.model_type} models on data with shape {X.shape}")

        if self.model is None:
            n_features = X.shape[1]
            n_classes = len(np.unique(y))
            self.create(n_features, n_classes)

        # Lưu các classes
        self.classes_ = np.unique(y)

        # Huấn luyện mô hình
        if self.is_tensorflow:
            # Chuyển đổi y thành one-hot encoding cho Keras
            y_categorical = to_categorical(y, num_classes=len(self.classes_))
            self.model.fit(X, y_categorical, epochs=self.params.get("epochs", 50),
                           batch_size=self.params.get("batch_size", 32),
                           verbose=1)
        else:
            self.model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán class labels cho dữ liệu.

        Args:
            X: Features

        Returns:
            y_pred: Class labels được dự đoán
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit() first.")

        if self.is_tensorflow:
            # Keras models
            y_pred_prob = self.model.predict(X)
            y_pred = np.argmax(y_pred_prob, axis=1)
        else:
            # Sklearn models
            y_pred = self.model.predict(X)

        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán probabilities cho mỗi class.

        Args:
            X: Features

        Returns:
            y_pred_prob: Probabilities của mỗi class
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit() first.")

        if self.is_tensorflow:
            # Keras models
            y_pred_prob = self.model.predict(X)
        else:
            # Sklearn models
            y_pred_prob = self.model.predict_proba(X)

        return y_pred_prob

    def evaluate(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> Dict[str, float]:
        """
        Đánh giá mô hình trên dữ liệu test.

        Args:
            X: Features
            y: Target labels thực tế

        Returns:
            metrics: Dictionary chứa các metrics đánh giá
        """
        # Dự đoán labels
        y_pred = self.predict(X)

        # Dự đoán probabilities
        y_pred_prob = self.predict_proba(X)

        # Tính toán metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="weighted"),
            "recall": recall_score(y, y_pred, average="weighted"),
            "f1": f1_score(y, y_pred, average="weighted"),
        }

        logger.info(f"Model evaluation: {metrics}")

        return metrics

    def save(self, path: str = None) -> str:
        """
        Lưu mô hình xuống file.

        Args:
            path: Đường dẫn để lưu mô hình. Nếu None, sẽ sử dụng đường dẫn mặc định.

        Returns:
            model_path: Đường dẫn đã lưu mô hình
        """
        if path is None:
            model_dir = settings.MODEL_PATH
            os.makedirs(model_dir, exist_ok=True)

            # Xác định đuôi file phù hợp
            extension = ".h5" if self.is_tensorflow else ".joblib"
            path = os.path.join(model_dir, f"{self.model_id}{extension}")

        logger.info(f"Saving models to {path}")

        # Lưu mô hình
        if self.is_tensorflow:
            # Lưu mô hình Keras
            self.model.save(path)
        else:
            # Lưu mô hình sklearn
            joblib.dump({
                "models": self.model,
                "classes_": self.classes_,
                "model_type": self.model_type,
                "params": self.params,
                "is_tensorflow": self.is_tensorflow
            }, path)

        return path

    @classmethod
    def load(cls, model_id: str = None, path: str = None) -> 'BaseModel':
        """
        Load mô hình từ file.

        Args:
            model_id: ID của mô hình. Nếu None, sẽ sử dụng model_id của path.
            path: Đường dẫn để load mô hình. Nếu None, sẽ sử dụng đường dẫn mặc định.

        Returns:
            models: BaseModel đã được load
        """
        if path is None and model_id is None:
            raise ValueError("Either model_id or path must be provided.")

        if path is None:
            model_dir = settings.MODEL_PATH

            # Thử tìm file với đuôi phù hợp
            joblib_path = os.path.join(model_dir, f"{model_id}.joblib")
            h5_path = os.path.join(model_dir, f"{model_id}.h5")

            if os.path.exists(joblib_path):
                path = joblib_path
            elif os.path.exists(h5_path):
                path = h5_path
            else:
                raise FileNotFoundError(f"No models file found for model_id: {model_id}")

        logger.info(f"Loading models from {path}")

        # Xác định loại mô hình dựa trên đuôi file
        is_tensorflow = path.endswith(".h5")

        # Load mô hình
        if is_tensorflow:
            # Load mô hình Keras
            keras_model = load_model(path)

            # Tạo instance mới
            model_instance = cls(model_type="neural_network", model_id=model_id)
            model_instance.model = keras_model
            model_instance.is_tensorflow = True

            # Xác định classes từ output shape
            output_shape = keras_model.output_shape[1]
            model_instance.classes_ = np.arange(output_shape)
        else:
            # Load mô hình sklearn
            saved_data = joblib.load(path)

            # Tạo instance mới
            model_instance = cls(
                model_type=saved_data["model_type"],
                params=saved_data["params"],
                model_id=model_id or os.path.splitext(os.path.basename(path))[0]
            )
            model_instance.model = saved_data["models"]
            model_instance.classes_ = saved_data["classes_"]
            model_instance.is_tensorflow = saved_data["is_tensorflow"]

        return model_instance