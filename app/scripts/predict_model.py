#!/usr/bin/env python
"""
Script để sử dụng mô hình đã huấn luyện dự đoán trên dữ liệu mới.
Sử dụng: python predict_model.py --model_path <đường_dẫn_đến_mô_hình> --data_path <đường_dẫn_đến_dữ_liệu> [--target_col <tên_cột_target>]
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.ml.inference.predict import ModelPredictor
    from app.ml.models.model import BaseModel
    from app.ml.data.preprocessing import DataPreprocessor
    from app.core.logging import setup_logging, Logger
except ImportError:
    # Nếu không thể import từ app, tạo các class giả lập cần thiết
    print("Không thể import từ module app. Sẽ tạo các class giả lập cần thiết.")
    import joblib

    # Cấu hình logging cơ bản
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


    class Logger:
        def __init__(self, name):
            self.logger = logging.getLogger(name)

        def info(self, msg, *args, **kwargs):
            self.logger.info(msg, *args, **kwargs)

        def error(self, msg, *args, **kwargs):
            self.logger.error(msg, *args, **kwargs)

        def warning(self, msg, *args, **kwargs):
            self.logger.warning(msg, *args, **kwargs)


    def setup_logging():
        pass


    class ModelPredictor:
        def __init__(self, model_id=None, model_path=None):
            self.model_id = model_id
            self.model_path = model_path
            self.model = None
            self.preprocessor = None
            self.logger = Logger(__name__)

            if model_path:
                self._load_model_from_path(model_path)

        def _load_model_from_path(self, model_path):
            """Load model and preprocessor from explicit paths."""
            try:
                # Xác định đường dẫn preprocessor từ model path
                model_dir = os.path.dirname(model_path)
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                preprocessor_path = os.path.join(model_dir, f"{model_name}_preprocessor.joblib")

                # Load model
                self.logger.info(f"Loading model from {model_path}")
                if model_path.endswith('.h5'):
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(model_path)
                    self.is_tensorflow = True
                else:  # .joblib
                    model_data = joblib.load(model_path)
                    self.model = model_data.get("model", model_data)
                    self.is_tensorflow = getattr(model_data, "is_tensorflow", False)

                # Load preprocessor
                self.logger.info(f"Loading preprocessor from {preprocessor_path}")
                if os.path.exists(preprocessor_path):
                    self.preprocessor = joblib.load(preprocessor_path)
                else:
                    self.logger.warning(f"Preprocessor not found at {preprocessor_path}")
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
                raise

        def predict(self, X):
            """
            Dự đoán class và probabilities cho dữ liệu đầu vào.
            """
            # Tiền xử lý dữ liệu
            if hasattr(self, 'preprocessor') and self.preprocessor is not None:
                if isinstance(X, pd.DataFrame):
                    X_processed = self.preprocessor.transform(X)
                else:
                    # Chuyển X thành DataFrame nếu cần
                    X_df = pd.DataFrame(X)
                    X_processed = self.preprocessor.transform(X_df)
            else:
                X_processed = X

            # Dự đoán
            if hasattr(self, 'is_tensorflow') and self.is_tensorflow:
                y_pred_proba = self.model.predict(X_processed)
                y_pred = np.argmax(y_pred_proba, axis=1)
                probabilities = {str(i): y_pred_proba[0][i] for i in range(len(y_pred_proba[0]))}
            else:
                y_pred = self.model.predict(X_processed)
                y_pred_proba = self.model.predict_proba(X_processed)
                probabilities = {str(i): y_pred_proba[0][i] for i in range(len(y_pred_proba[0]))}

            # Get class_name from class_mapping if available
            class_id = int(y_pred[0])
            if hasattr(self.preprocessor, 'class_mapping') and self.preprocessor.class_mapping:
                class_name = self.preprocessor.class_mapping.get(class_id, str(class_id))
            else:
                class_name = str(class_id)

            return class_id, class_name, probabilities

        def predict_batch(self, X):
            """
            Dự đoán classes và probabilities cho một batch dữ liệu.
            """
            # Tiền xử lý dữ liệu
            if hasattr(self, 'preprocessor') and self.preprocessor is not None:
                if isinstance(X, pd.DataFrame):
                    X_processed = self.preprocessor.transform(X)
                else:
                    # Chuyển X thành DataFrame nếu cần
                    X_df = pd.DataFrame(X)
                    X_processed = self.preprocessor.transform(X_df)
            else:
                X_processed = X

            # Dự đoán
            if hasattr(self, 'is_tensorflow') and self.is_tensorflow:
                y_pred_proba = self.model.predict(X_processed)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = self.model.predict(X_processed)
                y_pred_proba = self.model.predict_proba(X_processed)

            # Get class names from class_mapping if available
            class_ids = [int(i) for i in y_pred]
            class_names = []
            probabilities_list = []

            # Process each prediction
            for i, (class_id, proba) in enumerate(zip(y_pred, y_pred_proba)):
                if hasattr(self.preprocessor, 'class_mapping') and self.preprocessor.class_mapping:
                    class_name = self.preprocessor.class_mapping.get(int(class_id), str(class_id))
                else:
                    class_name = str(int(class_id))
                class_names.append(class_name)

                # Create probabilities dictionary
                if hasattr(self, 'is_tensorflow') and self.is_tensorflow:
                    prob_dict = {str(j): float(prob_val) for j, prob_val in enumerate(proba)}
                else:
                    prob_dict = {str(j): float(prob_val) for j, prob_val in enumerate(proba)}
                probabilities_list.append(prob_dict)

            return class_ids, class_names, probabilities_list


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Dự đoán sử dụng mô hình đã huấn luyện')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Đường dẫn đến file mô hình (joblib hoặc h5)')

    parser.add_argument('--data_path', type=str, required=True,
                        help='Đường dẫn đến file dữ liệu (CSV, XLSX, Parquet)')

    parser.add_argument('--target_col', type=str, default=None,
                        help='Tên cột target để đánh giá độ chính xác. Nếu không cung cấp, sẽ lấy cột cuối cùng')

    parser.add_argument('--feature_cols', type=str, nargs='+', default=None,
                        help='Danh sách các cột input được sử dụng để dự đoán. Nếu không cung cấp, sẽ sử dụng tất cả các cột trừ target')

    parser.add_argument('--output_path', type=str, default=None,
                        help='Đường dẫn để lưu kết quả dự đoán. Nếu không cung cấp, kết quả chỉ hiển thị trên console')

    parser.add_argument('--verbose', action='store_true',
                        help='In thông tin chi tiết trong quá trình dự đoán')

    return parser.parse_args()


def load_data(data_path, target_col=None, feature_cols=None):
    """
    Load dữ liệu từ file và tách features và target.

    Args:
        data_path: Đường dẫn đến file dữ liệu
        target_col: Tên cột target. Nếu None, mặc định là cột cuối cùng.
        feature_cols: Danh sách các cột được sử dụng làm features. Nếu None, sẽ sử dụng tất cả các cột trừ target.

    Returns:
        X: DataFrame chứa features
        y: Series chứa target (nếu có)
    """
    # Load dữ liệu
    try:
        if data_path.endswith('.csv'):
            try:
                df = pd.read_csv(data_path)
            except Exception:
                # Thử với delimiter khác nếu không đọc được
                df = pd.read_csv(data_path, sep=';')
        elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
            df = pd.read_excel(data_path)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError("Unsupported file format. Supported formats: .csv, .xlsx, .xls, .parquet")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

    # In tên các cột trong dataset để hỗ trợ người dùng
    print(f"Các cột trong dataset: {', '.join(df.columns)}")

    # Xác định target column
    if target_col is None:
        target_col = df.columns[-1]
        print(f"Sử dụng cột cuối cùng '{target_col}' làm target column.")

    # Xác định các cột feature
    if feature_cols is not None:
        # Kiểm tra các cột feature có tồn tại trong dữ liệu không
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Các cột feature không tồn tại trong dữ liệu: {', '.join(missing_cols)}")

        # Chỉ giữ lại các cột được chỉ định
        X = df[feature_cols]
        print(f"Sử dụng {len(feature_cols)} cột được chỉ định làm features.")
    else:
        # Kiểm tra nếu target column không tồn tại trong dữ liệu
        if target_col not in df.columns:
            print(f"Warning: Target column '{target_col}' not found. Predicting without evaluation metrics.")
            return df, None

        # Sử dụng tất cả các cột trừ target
        X = df.drop(columns=[target_col])
        print(f"Sử dụng tất cả {X.shape[1]} cột trừ target làm features.")

    # Tách target nếu có
    if target_col in df.columns:
        y = df[target_col]
    else:
        y = None
        print(f"Warning: Target column '{target_col}' not found. Predicting without evaluation metrics.")

    return X, y


def evaluate_predictions(y_true, y_pred, class_names=None):
    """
    Đánh giá kết quả dự đoán.

    Args:
        y_true: Labels thực tế
        y_pred: Labels dự đoán
        class_names: Tên các lớp

    Returns:
        metrics: Dictionary chứa các metrics đánh giá
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    # Tính confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return metrics, cm


def plot_confusion_matrix(cm, class_names, output_path=None):
    """
    Vẽ confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: Tên các lớp
        output_path: Đường dẫn để lưu plot. Nếu None, sẽ hiển thị plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_predictions_to_csv(X, y_true, y_pred, class_names, probabilities, output_path):
    """
    Lưu kết quả dự đoán vào file CSV.

    Args:
        X: DataFrame chứa features
        y_true: Series chứa labels thực tế
        y_pred: array chứa labels dự đoán
        class_names: List tên các lớp dự đoán
        probabilities: List dictionaries chứa probabilities
        output_path: Đường dẫn để lưu file
    """
    # Tạo DataFrame kết quả
    results_df = X.copy()

    # Thêm cột target thực tế nếu có
    if y_true is not None:
        results_df['true_label'] = y_true

    # Thêm cột dự đoán
    results_df['predicted_label'] = y_pred
    results_df['predicted_class_name'] = class_names

    # Thêm cột probabilities
    for i, prob_dict in enumerate(probabilities):
        for class_id, prob in prob_dict.items():
            results_df.loc[i, f'prob_class_{class_id}'] = prob

    # Lưu xuống file
    results_df.to_csv(output_path, index=False)


def main():
    """Main function to run the prediction process."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    try:
        setup_logging()
    except:
        # Fallback logging configuration if setup_logging fails
        logging.basicConfig(
            level=logging.INFO if args.verbose else logging.WARNING,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    logger = Logger(__name__)

    # Print start message
    logger.info("=== Bắt đầu quá trình dự đoán ===")
    start_time = datetime.now()

    try:
        # Load dữ liệu
        logger.info(f"Loading data from {args.data_path}")
        X, y = load_data(args.data_path, args.target_col)
        logger.info(f"Data loaded. Shape: {X.shape}")

        # Khởi tạo predictor với model path
        logger.info(f"Loading model from {args.model_path}")
        predictor = ModelPredictor(model_path=args.model_path)

        # Perform batch prediction
        logger.info("Performing predictions")
        class_ids, class_names_pred, probabilities = predictor.predict_batch(X)

        # Calculate evaluation metrics if target is available
        if y is not None:
            logger.info("Calculating evaluation metrics")
            metrics, cm = evaluate_predictions(y, class_ids)

            # Get unique classes for confusion matrix
            unique_classes = sorted(list(set(np.concatenate([np.unique(y), np.unique(class_ids)]))))

            # Get class names if available in preprocessor
            if hasattr(predictor.preprocessor, 'class_mapping') and predictor.preprocessor.class_mapping:
                class_names_for_cm = [predictor.preprocessor.class_mapping.get(i, str(i)) for i in unique_classes]
            else:
                class_names_for_cm = [str(c) for c in unique_classes]

            # Print results
            print("\n=== Kết quả đánh giá ===")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-score: {metrics['f1']:.4f}")

            # Show confusion matrix
            print("\nConfusion Matrix:")
            print(cm)

            # Plot confusion matrix if not in a headless environment
            try:
                if args.output_path:
                    cm_output_path = os.path.splitext(args.output_path)[0] + "_cm.png"
                    plot_confusion_matrix(cm, class_names_for_cm, cm_output_path)
                    logger.info(f"Confusion matrix saved to {cm_output_path}")
                else:
                    plot_confusion_matrix(cm, class_names_for_cm)
            except:
                logger.warning("Could not plot confusion matrix. May be running in a headless environment.")

        # Print some sample predictions
        print("\n=== Mẫu dự đoán ===")
        sample_size = min(5, len(class_ids))
        for i in range(sample_size):
            print(f"Sample {i + 1}:")
            print(f"  Predicted class: {class_names_pred[i]} (ID: {class_ids[i]})")

            # Print true label if available
            if y is not None:
                true_label = y.iloc[i]
                # Get true class name if available
                if hasattr(predictor.preprocessor, 'class_mapping') and predictor.preprocessor.class_mapping:
                    true_class_name = predictor.preprocessor.class_mapping.get(true_label, str(true_label))
                else:
                    true_class_name = str(true_label)
                print(f"  True class: {true_class_name} (ID: {true_label})")

                # Print if prediction is correct
                is_correct = class_ids[i] == true_label
                print(f"  Correct prediction: {'✓' if is_correct else '✗'}")

            # Print top 3 probabilities
            sorted_probs = sorted([(k, v) for k, v in probabilities[i].items()],
                                  key=lambda x: x[1], reverse=True)[:3]
            print("  Top probabilities:")
            for class_id, prob in sorted_probs:
                if hasattr(predictor.preprocessor, 'class_mapping') and predictor.preprocessor.class_mapping:
                    class_name = predictor.preprocessor.class_mapping.get(int(class_id), class_id)
                else:
                    class_name = class_id
                print(f"    {class_name}: {prob:.4f}")
            print()

        # Save predictions if output path is provided
        if args.output_path:
            save_predictions_to_csv(X, y, class_ids, class_names_pred, probabilities, args.output_path)
            logger.info(f"Predictions saved to {args.output_path}")

        # Calculate elapsed time
        end_time = datetime.now()
        prediction_time = end_time - start_time
        logger.info(f"Total prediction time: {prediction_time}")

        return {
            "predictions": {
                "class_ids": class_ids,
                "class_names": class_names_pred
            },
            "metrics": metrics if y is not None else None
        }

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()