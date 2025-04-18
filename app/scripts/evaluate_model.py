#!/usr/bin/env python
"""
Script để đánh giá mô hình phân loại đa lớp.
Sử dụng: python scripts/evaluate_model.py --data_path <đường_dẫn_đến_dữ_liệu> --model_id <id_của_mô_hình> [--output_path <đường_dẫn_output>]
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import json
from datetime import datetime

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.inference.predict import ModelPredictor
from app.ml.models.model import BaseModel
from app.ml.data.preprocessing import DataPreprocessor
from app.core.logging import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a multi-class classification model')

    parser.add_argument('--data_path', type=str, required=True,
                        help='Đường dẫn đến file dữ liệu (CSV, XLSX, Parquet)')

    parser.add_argument('--model_id', type=str, default=None,
                        help='ID của mô hình cần đánh giá. Nếu không cung cấp, sẽ sử dụng mô hình mới nhất.')

    parser.add_argument('--output_path', type=str, default=None,
                        help='Đường dẫn để lưu báo cáo đánh giá. Nếu không cung cấp, báo cáo sẽ chỉ được in ra console.')

    parser.add_argument('--target_col', type=str, default=None,
                        help='Tên cột target trong dữ liệu. Nếu không cung cấp, mặc định là cột cuối cùng.')

    parser.add_argument('--verbose', action='store_true',
                        help='In thông tin chi tiết trong quá trình đánh giá')

    return parser.parse_args()


def load_data(data_path, target_col=None):
    """
    Load dữ liệu từ file và tách features và target.

    Args:
        data_path: Đường dẫn đến file dữ liệu
        target_col: Tên cột target. Nếu None, mặc định là cột cuối cùng.

    Returns:
        X: DataFrame chứa features
        y: Series chứa target
    """
    # Load dữ liệu
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        df = pd.read_excel(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported file format. Supported formats: .csv, .xlsx, .xls, .parquet")

    # Xác định target column
    if target_col is None:
        target_col = df.columns[-1]

    # Tách features và target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def plot_confusion_matrix(cm, class_names, output_path=None):
    """
    Vẽ confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: Tên các lớp
        output_path: Đường dẫn để lưu plot. Nếu None, sẽ chỉ hiển thị plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    """Main function to run the evaluation process."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging()
    log_level = logging.INFO if args.verbose else logging.WARNING
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Print start message
    logger.info("=== Bắt đầu quá trình đánh giá mô hình ===")
    start_time = datetime.now()

    try:
        # Load data
        logger.info(f"Loading data from {args.data_path}")
        X, y = load_data(args.data_path, args.target_col)
        logger.info(f"Data loaded. Shape: {X.shape}")

        # Initialize predictor with the specified model
        logger.info(f"Loading model {args.model_id if args.model_id else 'latest'}")
        predictor = ModelPredictor(model_id=args.model_id)

        # Get unique class values and their names
        unique_classes = np.unique(y)
        if hasattr(predictor.preprocessor, 'class_mapping') and predictor.preprocessor.class_mapping:
            class_names = [predictor.preprocessor.class_mapping.get(i, str(i)) for i in range(len(unique_classes))]
        else:
            class_names = [str(c) for c in unique_classes]

        # Perform batch prediction
        logger.info("Performing predictions")
        class_ids, class_names_pred, probabilities = predictor.predict_batch(X)
        y_pred = np.array(class_ids)

        # Calculate metrics
        logger.info("Calculating evaluation metrics")
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='weighted')

        # Generate classification report
        clf_report = classification_report(y, y_pred, target_names=class_names, output_dict=True)

        # Generate confusion matrix
        cm = confusion_matrix(y, y_pred)

        # Print results
        logger.info(f"Evaluation completed for model: {predictor.model_id}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-score: {f1:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y, y_pred, target_names=class_names))

        # Calculate elapsed time
        end_time = datetime.now()
        evaluation_time = end_time - start_time
        logger.info(f"Total evaluation time: {evaluation_time}")

        # Save results if output path is provided
        if args.output_path:
            output_dir = os.path.dirname(args.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Determine base output path without extension
            base_output_path = os.path.splitext(args.output_path)[0]

            # Save metrics as JSON
            metrics = {
                "model_id": predictor.model_id,
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "classification_report": clf_report,
                "evaluation_time": str(evaluation_time),
                "data_path": args.data_path,
                "data_shape": X.shape,
                "evaluation_date": datetime.now().isoformat()
            }

            with open(f"{base_output_path}.json", 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Evaluation metrics saved to {base_output_path}.json")

            # Save confusion matrix plot
            plot_confusion_matrix(cm, class_names, f"{base_output_path}_cm.png")
            logger.info(f"Confusion matrix plot saved to {base_output_path}_cm.png")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "classification_report": clf_report
        }

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()