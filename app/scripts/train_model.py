#!/usr/bin/env python
"""
Script để huấn luyện mô hình phân loại đa lớp.
Sử dụng: python scripts/train_model.py --data_path <đường_dẫn_đến_dữ_liệu> --config_path <đường_dẫn_đến_config> [--model_type <loại_mô_hình>]
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
from datetime import datetime

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.models.train import ModelTrainer
from app.core.config import settings, get_model_config
from app.core.logging import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a multi-class classification model')

    parser.add_argument('--data_path', type=str, required=True,
                        help='Đường dẫn đến file dữ liệu (CSV, XLSX, Parquet)')

    parser.add_argument('--config_path', type=str, default=None,
                        help='Đường dẫn đến file config. Nếu không cung cấp, sẽ sử dụng config mặc định.')

    parser.add_argument('--model_type', type=str, default=None,
                        help='Loại mô hình (random_forest, svm, neural_network). Nếu không cung cấp, sẽ sử dụng loại từ config.')

    parser.add_argument('--output_dir', type=str, default=None,
                        help='Thư mục để lưu mô hình. Nếu không cung cấp, sẽ sử dụng đường dẫn mặc định.')

    parser.add_argument('--verbose', action='store_true',
                        help='In thông tin chi tiết trong quá trình huấn luyện')

    return parser.parse_args()


def main():
    """Main function to run the training process."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging()
    log_level = logging.INFO if args.verbose else logging.WARNING
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Print start message
    logger.info("=== Bắt đầu quá trình huấn luyện mô hình ===")
    start_time = datetime.now()

    # Load config if provided
    config = None
    if args.config_path:
        try:
            with open(args.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {args.config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            sys.exit(1)

    # Override model type if provided
    if args.model_type and config:
        if "model" not in config:
            config["model"] = {}
        config["model"]["type"] = args.model_type
        logger.info(f"Overriding model type to {args.model_type}")
    elif args.model_type:
        config = {"model": {"type": args.model_type}}
        logger.info(f"Using model type {args.model_type}")

    # Override output directory if provided
    if args.output_dir:
        os.environ["MODEL_PATH"] = args.output_dir
        logger.info(f"Setting model output directory to {args.output_dir}")

    # Initialize trainer
    trainer = ModelTrainer(config)

    try:
        # Train model
        logger.info(f"Training model using data from {args.data_path}")
        model_id, metrics = trainer.train(data_path=args.data_path)

        # Print results
        logger.info(f"Model training completed successfully. Model ID: {model_id}")
        logger.info(f"Evaluation metrics: {metrics}")

        # Print training time
        end_time = datetime.now()
        training_time = end_time - start_time
        logger.info(f"Total training time: {training_time}")

        # Print model location
        model_path = os.path.join(settings.MODEL_PATH, f"{model_id}.joblib")
        preprocessor_path = os.path.join(settings.MODEL_PATH, f"{model_id}_preprocessor.joblib")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Preprocessor saved to: {preprocessor_path}")

        return model_id, metrics

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()