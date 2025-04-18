import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List, Any

from app.core.config import settings
from app.core.logging import Logger
from app.ml.models.model import BaseModel
from app.ml.data.preprocessing import DataPreprocessor

logger = Logger(__name__)


class ModelPredictor:
    """
    Class quản lý quá trình dự đoán sử dụng mô hình đã huấn luyện.
    """

    def __init__(self, model_id: str = None):
        """
        Khởi tạo predictor với một mô hình cụ thể.

        Args:
            model_id: ID của mô hình cần sử dụng. Nếu None, sẽ sử dụng mô hình mới nhất.
        """
        self.model_id = model_id
        self.model = None
        self.preprocessor = None

        # Load mô hình
        self._load_model()

        logger.info(f"Initialized ModelPredictor with model: {self.model_id}")

    def _load_model(self) -> None:
        """
        Load mô hình và preprocessor.
        """
        model_dir = settings.MODEL_PATH

        # Nếu không chỉ định model_id, tìm mô hình mới nhất
        if self.model_id is None:
            # Lấy tất cả các file models (joblib hoặc h5)
            model_files = []
            for file in os.listdir(model_dir):
                if (file.endswith(".joblib") or file.endswith(".h5")) and not file.endswith("_preprocessor.joblib"):
                    model_files.append(file)

            if not model_files:
                raise FileNotFoundError("No models found.")

            # Sắp xếp theo thời gian tạo để lấy mô hình mới nhất
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
            self.model_id = os.path.splitext(model_files[0])[0]

        # Load mô hình
        try:
            self.model = BaseModel.load(model_id=self.model_id)
            logger.info(f"Loaded model: {self.model_id}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

        # Load preprocessor
        try:
            preprocessor_path = os.path.join(model_dir, f"{self.model_id}_preprocessor.joblib")
            self.preprocessor = DataPreprocessor.load(preprocessor_path)
            logger.info(f"Loaded preprocessor for model: {self.model_id}")
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            raise

    def predict(
            self,
            X: Union[np.ndarray, pd.DataFrame, List]
    ) -> Tuple[int, str, Dict[str, float]]:
        """
        Dự đoán class và probabilities cho dữ liệu đầu vào.

        Args:
            X: Dữ liệu đầu vào (có thể là numpy array, pandas DataFrame, hoặc list)

        Returns:
            class_id: ID của class được dự đoán
            class_name: Tên của class được dự đoán
            probabilities: Dictionary chứa probabilities cho mỗi class
        """
        logger.info(f"Making prediction for input with shape {np.array(X).shape}")

        # Chuyển đổi input thành numpy array
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        elif isinstance(X, list):
            X_array = np.array(X)
        else:
            X_array = X

        # Reshape nếu cần
        if len(X_array.shape) == 1:
            X_array = X_array.reshape(1, -1)

        # Tiền xử lý dữ liệu
        if self.preprocessor is not None:
            # Chuyển đổi thành DataFrame nếu cần
            if not isinstance(X_array, pd.DataFrame):
                # Tạo DataFrame với tên cột mặc định
                X_df = pd.DataFrame(X_array)
            else:
                X_df = X_array

            # Tiền xử lý
            X_processed = self.preprocessor.transform(X_df)
        else:
            X_processed = X_array

        # Dự đoán
        try:
            # Dự đoán class
            y_pred = self.model.predict(X_processed)

            # Dự đoán probabilities
            y_pred_proba = self.model.predict_proba(X_processed)

            # Lấy class ID và chuyển đổi thành int để serialization
            class_id = int(y_pred[0])

            # Lấy class name từ class mapping nếu có
            if hasattr(self.preprocessor, 'class_mapping') and self.preprocessor.class_mapping:
                class_name = self.preprocessor.class_mapping.get(class_id, str(class_id))
            else:
                class_name = str(class_id)

            # Dictionary probabilities cho mỗi class
            probabilities = {}
            for i, prob in enumerate(y_pred_proba[0]):
                if hasattr(self.preprocessor, 'class_mapping') and self.preprocessor.class_mapping:
                    class_name_i = self.preprocessor.class_mapping.get(i, str(i))
                else:
                    class_name_i = str(i)
                probabilities[class_name_i] = float(prob)

            logger.info(f"Prediction successful. Class: {class_name} (ID: {class_id})")

            return class_id, class_name, probabilities

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def predict_batch(
            self,
            X: Union[np.ndarray, pd.DataFrame, List]
    ) -> Tuple[List[int], List[str], List[Dict[str, float]]]:
        """
        Dự đoán classes và probabilities cho một batch dữ liệu.

        Args:
            X: Batch dữ liệu đầu vào

        Returns:
            class_ids: List các ID của classes được dự đoán
            class_names: List các tên của classes được dự đoán
            probabilities_list: List các dictionaries chứa probabilities
        """
        logger.info(f"Making batch prediction for input with shape {np.array(X).shape}")

        # Chuyển đổi input thành numpy array
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        elif isinstance(X, list):
            X_array = np.array(X)
        else:
            X_array = X

        # Tiền xử lý dữ liệu
        if self.preprocessor is not None:
            # Chuyển đổi thành DataFrame nếu cần
            if not isinstance(X_array, pd.DataFrame):
                # Tạo DataFrame với tên cột mặc định
                X_df = pd.DataFrame(X_array)
            else:
                X_df = X_array

            # Tiền xử lý
            X_processed = self.preprocessor.transform(X_df)
        else:
            X_processed = X_array

        # Dự đoán
        try:
            # Dự đoán class
            y_pred = self.model.predict(X_processed)

            # Dự đoán probabilities
            y_pred_proba = self.model.predict_proba(X_processed)

            class_ids = []
            class_names = []
            probabilities_list = []

            # Xử lý từng dự đoán
            for i, (class_id, proba) in enumerate(zip(y_pred, y_pred_proba)):
                # Chuyển đổi class_id thành int
                class_id_int = int(class_id)
                class_ids.append(class_id_int)

                # Lấy class name
                if hasattr(self.preprocessor, 'class_mapping') and self.preprocessor.class_mapping:
                    class_name = self.preprocessor.class_mapping.get(class_id_int, str(class_id_int))
                else:
                    class_name = str(class_id_int)
                class_names.append(class_name)

                # Dictionary probabilities
                probabilities = {}
                for j, prob in enumerate(proba):
                    if hasattr(self.preprocessor, 'class_mapping') and self.preprocessor.class_mapping:
                        class_name_j = self.preprocessor.class_mapping.get(j, str(j))
                    else:
                        class_name_j = str(j)
                    probabilities[class_name_j] = float(prob)

                probabilities_list.append(probabilities)

            logger.info(f"Batch prediction successful. Processed {len(class_ids)} samples.")

            return class_ids, class_names, probabilities_list

        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise