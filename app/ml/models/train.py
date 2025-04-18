import os
import uuid
from typing import Dict, Tuple, Union, List, Any
import numpy as np
import pandas as pd
import json
from datetime import datetime

from app.core.config import settings, get_model_config
from app.core.logging import Logger
from app.ml.data.preprocessing import DataPreprocessor
from app.ml.data.validation import DataValidator
from app.ml.models.model import BaseModel

logger = Logger(__name__)


class ModelTrainer:
    """
    Class quản lý quá trình huấn luyện mô hình.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Khởi tạo trainer với cấu hình.

        Args:
            config: Cấu hình cho việc training. Nếu None, sẽ load từ file cấu hình.
        """
        if config is None:
            self.config = get_model_config()
        else:
            self.config = config

        self.model_type = self.config.get("models", {}).get("type", "random_forest")
        self.model_params = self.config.get("models", {}).get("params", {}).get(self.model_type, {})

        logger.info(f"Initialized ModelTrainer with models type: {self.model_type}")

    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load dữ liệu từ file.

        Args:
            data_path: Đường dẫn đến file dữ liệu

        Returns:
            X: DataFrame chứa features
            y: Series chứa target labels
        """
        logger.info(f"Loading data from {data_path}")

        try:
            # Load dữ liệu
            if data_path.endswith(".csv"):
                df = pd.read_csv(data_path, sep=';')
            elif data_path.endswith(".parquet"):
                df = pd.read_parquet(data_path)
            elif data_path.endswith((".xls", ".xlsx")):
                df = pd.read_excel(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")

            # Lấy tên target column từ config hoặc sử dụng mặc định là column cuối cùng
            target_col = self.config.get("training", {}).get("target_column")

            if target_col is None:
                # Mặc định sử dụng column cuối cùng làm target
                target_col = df.columns[-1]

            # Tách features và target
            X = df.drop(columns=[target_col])
            y = df[target_col]

            logger.info(f"Data loaded successfully with shape {X.shape}")

            return X, y

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def train(
            self,
            data_path: str = None,
            X: pd.DataFrame = None,
            y: pd.Series = None,
            model_params: Dict[str, Any] = None
    ) -> Tuple[str, Dict[str, float]]:
        """
        Huấn luyện mô hình trên dữ liệu.

        Args:
            data_path: Đường dẫn đến file dữ liệu (bỏ qua nếu X và y được cung cấp)
            X: DataFrame chứa features (bỏ qua nếu data_path được cung cấp)
            y: Series chứa target labels (bỏ qua nếu data_path được cung cấp)
            model_params: Override cho các tham số mô hình

        Returns:
            model_id: ID của mô hình đã huấn luyện
            metrics: Dictionary chứa các metrics đánh giá
        """
        # Ghi đè tham số mô hình nếu được cung cấp
        if model_params is not None:
            self.model_params.update(model_params)

        # Load dữ liệu nếu cần
        if X is None or y is None:
            if data_path is None:
                raise ValueError("Either (X, y) or data_path must be provided.")
            X, y = self.load_data(data_path)

        # Tạo models ID duy nhất
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{self.model_type}_{timestamp}"

        logger.info(f"Starting training process for models: {model_id}")

        # Tiền xử lý dữ liệu - cập nhật config để bao gồm cấu hình Vietnamese processing
        preprocessor_config = self.config.get("preprocessing", {})
        # Kiểm tra xem có cần xử lý tiếng Việt và mã số thuế không
        preprocessor_config["vietnamese_processing"] = self.config.get("preprocessing", {}).get("vietnamese_processing",
                                                                                                False)
        preprocessor_config["tax_id_processing"] = self.config.get("preprocessing", {}).get("tax_id_processing", False)

        preprocessor = DataPreprocessor(preprocessor_config)
        validator = DataValidator()

        # Phân chia dữ liệu
        X_train, X_val, X_test, y_train, y_val, y_test = validator.train_test_validation_split(X, y)

        # Fit và transform dữ liệu train
        X_train_processed = preprocessor.fit_transform(X_train, y_train)

        # Transform dữ liệu validation và test
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)

        # Transform target nếu cần
        y_train_processed = preprocessor.transform_target(y_train)
        y_val_processed = preprocessor.transform_target(y_val)
        y_test_processed = preprocessor.transform_target(y_test)

        # Kiểm tra chất lượng dữ liệu
        data_quality = validator.check_data_quality(X_train_processed, y_train_processed)
        logger.info(f"Data quality check: {json.dumps(data_quality, default=str)}")

        # Khởi tạo mô hình
        model = BaseModel(model_type=self.model_type, params=self.model_params, model_id=model_id)

        # Huấn luyện mô hình
        model.create(n_features=X_train_processed.shape[1], n_classes=len(np.unique(y_train_processed)))
        model.fit(X_train_processed, y_train_processed)

        # Đánh giá mô hình trên validation set
        val_metrics = model.evaluate(X_val_processed, y_val_processed)
        logger.info(f"Validation metrics: {val_metrics}")

        # Đánh giá mô hình trên test set
        test_metrics = model.evaluate(X_test_processed, y_test_processed)
        logger.info(f"Test metrics: {test_metrics}")

        # Lưu mô hình và preprocessor
        model_path = model.save()
        preprocessor_path = os.path.join(settings.MODEL_PATH, f"{model_id}_preprocessor.joblib")
        preprocessor.save(preprocessor_path)

        # Lưu metadata về mô hình
        metadata = {
            "model_id": model_id,
            "model_type": self.model_type,
            "created_at": datetime.now().isoformat(),
            "data_shape": X.shape,
            "params": self.model_params,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "data_quality": data_quality
        }

        metadata_path = os.path.join(settings.MODEL_PATH, f"{model_id}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4, default=str)

        logger.info(f"Model {model_id} training completed and saved to {model_path}")

        return model_id, test_metrics