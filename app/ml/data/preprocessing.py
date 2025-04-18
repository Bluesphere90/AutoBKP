# app/ml/data/preprocessing.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
import re
import unicodedata

from app.core.config import settings, get_model_config
from app.core.logging import Logger
from app.ml.data.vietnamese_preprocessing import DataProcessor  # Import DataProcessor mới

logger = Logger(__name__)


class DataPreprocessor:
    """
    Class xử lý dữ liệu cho bài toán multi-class classification.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Khởi tạo preprocessor với cấu hình.

        Args:
            config: Cấu hình cho preprocessing. Nếu None, sẽ load từ file cấu hình.
        """
        if config is None:
            model_config = get_model_config()
            self.config = model_config.get("preprocessing", {})
        else:
            self.config = config

        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.class_mapping = {}

        # Thêm flag xử lý tiếng Việt và mã số thuế
        self.vietnamese_processing = self.config.get("vietnamese_processing", False)
        self.tax_id_processing = self.config.get("tax_id_processing", False)

        # Khởi tạo Vietnamese Processor nếu cần
        if self.vietnamese_processing:
            self.vietnamese_processor = DataProcessor()

        self.scaling_method = self.config.get("scaling", "standard")
        self.encoding_method = self.config.get("categorical_encoding", "one_hot")
        self.missing_values_method = self.config.get("missing_values", "mean")

        logger.info(f"Initialized DataPreprocessor with config: {self.config}")

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'DataPreprocessor':
        """
        Fit transformers trên dữ liệu.

        Args:
            X: DataFrame chứa features
            y: Series chứa target labels (optional)

        Returns:
            self: preprocessor đã fit
        """
        logger.info(f"Fitting preprocessor on data with shape {X.shape}")

        # Xử lý dữ liệu tiếng Việt và mã số thuế nếu cần
        if self.vietnamese_processing:
            logger.info("Applying Vietnamese data processing")
            # Lưu lại tên các cột
            self.original_columns = X.columns.tolist()

            # Tìm và lưu tên cột target nếu có
            target_col_name = None
            if y is not None and isinstance(y, pd.Series) and y.name:
                target_col_name = y.name

            # Tạo DataFrame kết hợp để xử lý
            if y is not None and target_col_name:
                combined_df = X.copy()
                combined_df[target_col_name] = y

                # Áp dụng Vietnamese processor
                processed_df = self.vietnamese_processor.fit_transform(combined_df, target_col=target_col_name)

                # Tách lại X và cập nhật các mapping
                X = processed_df.drop(columns=[target_col_name, f"{target_col_name}_encoded"])

                # Lưu lại target mapping nếu có
                if hasattr(self.vietnamese_processor, 'target_mapping') and self.vietnamese_processor.target_mapping:
                    self.class_mapping = self.vietnamese_processor.target_mapping

                # Sau bước xử lý tiếng Việt, các bước xử lý khác vẫn sẽ tiếp tục
            else:
                # Xử lý chỉ với X nếu không có target
                X = self.vietnamese_processor.fit_transform(X)

        # Xử lý missing values
        self._fit_imputers(X)

        # Feature scaling
        self._fit_scalers(X)

        # Encoding categorical features (nếu chưa được xử lý bởi Vietnamese processor)
        if not self.vietnamese_processing:
            self._fit_encoders(X)

        # Label encoding cho target nếu cần và chưa được xử lý
        if y is not None and not self.vietnamese_processing:
            self._fit_target_encoder(y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dữ liệu sử dụng các transformers đã fit.

        Args:
            X: DataFrame chứa features

        Returns:
            X_transformed: DataFrame đã được transform
        """
        logger.info(f"Transforming data with shape {X.shape}")

        # Copy để tránh thay đổi dữ liệu gốc
        X_transformed = X.copy()

        # Xử lý dữ liệu tiếng Việt và mã số thuế nếu cần
        if self.vietnamese_processing:
            logger.info("Applying Vietnamese data transformation")
            X_transformed = self.vietnamese_processor.transform(X_transformed)

            # Sau bước xử lý tiếng Việt, các bước xử lý khác vẫn sẽ tiếp tục nếu có columns cần xử lý thêm
            # (như các cột số chưa được scale)

        # Xử lý missing values
        X_transformed = self._transform_imputers(X_transformed)

        # Feature scaling
        X_transformed = self._transform_scalers(X_transformed)

        # Encoding categorical features nếu chưa được xử lý
        if not self.vietnamese_processing:
            X_transformed = self._transform_encoders(X_transformed)

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit và transform dữ liệu.

        Args:
            X: DataFrame chứa features
            y: Series chứa target labels (optional)

        Returns:
            X_transformed: DataFrame đã được transform
        """
        return self.fit(X, y).transform(X)

    def transform_target(self, y: pd.Series) -> np.ndarray:
        """
        Transform target labels.

        Args:
            y: Series chứa target labels

        Returns:
            y_transformed: array đã được transform
        """
        if self.encoding_method == "label" and "target" in self.encoders:
            return self.encoders["target"].transform(y)
        return y.values

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Chuyển target labels trở lại dạng ban đầu.

        Args:
            y: array chứa encoded labels

        Returns:
            y_original: array dạng ban đầu
        """
        if self.encoding_method == "label" and "target" in self.encoders:
            return self.encoders["target"].inverse_transform(y)
        return y

    def _fit_imputers(self, X: pd.DataFrame) -> None:
        """Fit imputers cho missing values."""
        numeric_cols = X.select_dtypes(include=np.number).columns
        categorical_cols = X.select_dtypes(include=['object']).columns

        # Imputer cho numeric columns
        if len(numeric_cols) > 0:
            self.imputers["numeric"] = SimpleImputer(
                strategy=self.missing_values_method
            )
            self.imputers["numeric"].fit(X[numeric_cols])

        # Imputer cho categorical columns
        if len(categorical_cols) > 0:
            self.imputers["categorical"] = SimpleImputer(
                strategy="most_frequent"
            )
            self.imputers["categorical"].fit(X[categorical_cols])

    def _transform_imputers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform dữ liệu sử dụng imputers."""
        X_transformed = X.copy()

        numeric_cols = X.select_dtypes(include=np.number).columns
        categorical_cols = X.select_dtypes(include=['object']).columns

        # Apply imputer cho numeric columns
        if len(numeric_cols) > 0 and "numeric" in self.imputers:
            X_transformed[numeric_cols] = self.imputers["numeric"].transform(X[numeric_cols])

        # Apply imputer cho categorical columns
        if len(categorical_cols) > 0 and "categorical" in self.imputers:
            X_transformed[categorical_cols] = self.imputers["categorical"].transform(X[categorical_cols])

        return X_transformed

    def _fit_scalers(self, X: pd.DataFrame) -> None:
        """Fit scalers cho numeric features."""
        numeric_cols = X.select_dtypes(include=np.number).columns

        if len(numeric_cols) == 0:
            return

        # Tạo scaler phù hợp
        if self.scaling_method == "standard":
            self.scalers["numeric"] = StandardScaler()
        elif self.scaling_method == "minmax":
            self.scalers["numeric"] = MinMaxScaler()
        elif self.scaling_method == "robust":
            self.scalers["numeric"] = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {self.scaling_method}. Using StandardScaler.")
            self.scalers["numeric"] = StandardScaler()

        # Fit scaler
        self.scalers["numeric"].fit(X[numeric_cols])

    def _transform_scalers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform dữ liệu sử dụng scalers."""
        X_transformed = X.copy()

        numeric_cols = X.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0 and "numeric" in self.scalers:
            scaled_features = self.scalers["numeric"].transform(X[numeric_cols])
            X_transformed[numeric_cols] = scaled_features

        return X_transformed

    def _fit_encoders(self, X: pd.DataFrame) -> None:
        """Fit encoders cho categorical features."""
        categorical_cols = X.select_dtypes(include=['object']).columns

        if len(categorical_cols) == 0:
            return

        if self.encoding_method == "one_hot":
            self.encoders["categorical"] = OneHotEncoder(sparse=False, handle_unknown="ignore")
            self.encoders["categorical"].fit(X[categorical_cols])
        elif self.encoding_method == "label":
            self.encoders["categorical"] = {}
            for col in categorical_cols:
                self.encoders["categorical"][col] = LabelEncoder()
                self.encoders["categorical"][col].fit(X[col].astype(str))
        else:
            logger.warning(f"Unknown encoding method: {self.encoding_method}. Using OneHotEncoder.")
            self.encoders["categorical"] = OneHotEncoder(sparse=False, handle_unknown="ignore")
            self.encoders["categorical"].fit(X[categorical_cols])

    def _transform_encoders(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform dữ liệu sử dụng encoders."""
        X_transformed = X.copy()

        categorical_cols = X.select_dtypes(include=['object']).columns

        if len(categorical_cols) == 0 or "categorical" not in self.encoders:
            return X_transformed

        if self.encoding_method == "one_hot":
            # One-hot encoding
            encoded_array = self.encoders["categorical"].transform(X[categorical_cols])
            feature_names = self.encoders["categorical"].get_feature_names_out(categorical_cols)

            # Thêm encoded features vào DataFrame
            encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=X.index)

            # Xóa categorical columns gốc và nối với encoded features
            X_transformed = X_transformed.drop(columns=categorical_cols)
            X_transformed = pd.concat([X_transformed, encoded_df], axis=1)

        elif self.encoding_method == "label":
            # Label encoding
            for col in categorical_cols:
                X_transformed[col] = self.encoders["categorical"][col].transform(X[col].astype(str))

        return X_transformed

    def _fit_target_encoder(self, y: pd.Series) -> None:
        """Fit encoder cho target labels."""
        self.encoders["target"] = LabelEncoder()
        self.encoders["target"].fit(y)

        # Lưu mapping từ index sang tên class
        self.class_mapping = {
            i: class_name for i, class_name in enumerate(self.encoders["target"].classes_)
        }

    def save(self, path: str = None) -> None:
        """
        Lưu preprocessor xuống file.

        Args:
            path: Đường dẫn để lưu preprocessor. Nếu None, sẽ sử dụng đường dẫn mặc định.
        """
        if path is None:
            path = os.path.join(settings.MODEL_PATH, "preprocessor.joblib")

        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Lưu preprocessor
        joblib.dump(self, path)
        logger.info(f"Saved preprocessor to {path}")

    @classmethod
    def load(cls, path: str = None) -> 'DataPreprocessor':
        """
        Load preprocessor từ file.

        Args:
            path: Đường dẫn để load preprocessor. Nếu None, sẽ sử dụng đường dẫn mặc định.

        Returns:
            preprocessor: DataPreprocessor đã được load
        """
        if path is None:
            path = os.path.join(settings.MODEL_PATH, "preprocessor.joblib")

        # Load preprocessor
        preprocessor = joblib.load(path)
        logger.info(f"Loaded preprocessor from {path}")

        return preprocessor