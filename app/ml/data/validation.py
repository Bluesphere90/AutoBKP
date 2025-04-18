import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List, Any
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import logging

from app.core.config import get_model_config
from app.core.logging import Logger

logger = Logger(__name__)


class DataValidator:
    """
    Class phục vụ validation dữ liệu và phân chia train/validation/test.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Khởi tạo validator với cấu hình.

        Args:
            config: Cấu hình cho validation. Nếu None, sẽ load từ file cấu hình.
        """
        if config is None:
            model_config = get_model_config()
            self.config = model_config.get("training", {})
        else:
            self.config = config

        self.test_size = self.config.get("test_size", 0.2)
        self.validation_size = self.config.get("validation_size", 0.1)
        self.random_state = self.config.get("random_state", 42)
        self.cv_folds = self.config.get("cross_validation_folds", 5)

        logger.info(f"Initialized DataValidator with config: {self.config}")

    def train_test_validation_split(
            self,
            X: pd.DataFrame,
            y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Phân chia dữ liệu thành train, validation và test sets.

        Args:
            X: DataFrame chứa features
            y: Series chứa target labels

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test: Các tập dữ liệu đã phân chia
        """
        logger.info(f"Splitting data with shape {X.shape} into train/val/test")

        # Tính tỉ lệ test trên tổng dữ liệu
        test_ratio = self.test_size

        # Tính tỉ lệ validation trên phần dữ liệu còn lại (không bao gồm test)
        val_ratio = self.validation_size / (1 - test_ratio)

        # Đầu tiên, tách dữ liệu thành train+val và test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=self.random_state,
            stratify=y
        )

        # Sau đó, tách dữ liệu train+val thành train và val
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=y_trainval
        )

        logger.info(f"Split completed. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_cross_validation_splits(
            self,
            X: pd.DataFrame,
            y: pd.Series
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Tạo các cross-validation folds.

        Args:
            X: DataFrame chứa features
            y: Series chứa target labels

        Returns:
            folds: List các tuples (train_indices, val_indices)
        """
        logger.info(f"Creating {self.cv_folds} cross-validation folds")

        # Sử dụng StratifiedKFold cho bài toán classification
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # Tạo các folds
        folds = list(skf.split(X, y))

        return folds

    def check_data_quality(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, Any]:
        """
        Kiểm tra chất lượng dữ liệu.

        Args:
            X: DataFrame chứa features
            y: Series chứa target labels (optional)

        Returns:
            report: Dictionary chứa thông tin về chất lượng dữ liệu
        """
        logger.info(f"Checking data quality for data with shape {X.shape}")

        report = {}

        # Kiểm tra missing values
        missing_values = X.isnull().sum()
        report["missing_values"] = {
            "total": missing_values.sum(),
            "by_column": missing_values[missing_values > 0].to_dict()
        }

        # Kiểm tra các outliers trong numeric columns
        numeric_cols = X.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            # Sử dụng IQR để xác định outliers
            Q1 = X[numeric_cols].quantile(0.25)
            Q3 = X[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1

            outliers = {}
            for col in numeric_cols:
                lower_bound = Q1[col] - 1.5 * IQR[col]
                upper_bound = Q3[col] + 1.5 * IQR[col]
                n_outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
                if n_outliers > 0:
                    outliers[col] = n_outliers

            report["outliers"] = outliers

        # Kiểm tra class imbalance nếu có y
        if y is not None:
            class_counts = y.value_counts()
            report["class_distribution"] = class_counts.to_dict()

            # Tính imbalance ratio (max/min)
            imbalance_ratio = class_counts.max() / class_counts.min()
            report["imbalance_ratio"] = imbalance_ratio

        # Kiểm tra cardinality của categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cardinality = {col: X[col].nunique() for col in categorical_cols}
            report["categorical_cardinality"] = cardinality

        return report