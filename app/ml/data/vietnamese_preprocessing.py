import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import os
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Class xử lý dữ liệu đầu vào cho bài toán phân loại với dữ liệu Tiếng Việt.
    """

    def __init__(self, config=None):
        """
        Khởi tạo processor với cấu hình.

        Args:
            config: Dict cấu hình cho việc tiền xử lý.
        """
        self.config = config or {}
        self.label_encoders = {}
        self.imputers = {}
        self.stats = {}

    def normalize_vietnamese_text(self, text):
        """
        Chuẩn hóa text Tiếng Việt: loại bỏ dấu câu, chuyển về chữ thường.
        """
        if not isinstance(text, str):
            return str(text) if pd.notna(text) else ""

        # Chuyển về chữ thường và loại bỏ dấu câu
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalize_tax_id(self, tax_id):
        """
        Chuẩn hóa mã số thuế: giữ nguyên định dạng, xử lý các trường hợp đặc biệt.
        """
        if pd.isna(tax_id):
            return ""

        # Chuyển thành string nếu là số
        tax_id = str(tax_id)

        # Loại bỏ khoảng trắng
        tax_id = tax_id.strip()

        # Loại bỏ các ký tự không phải số và dấu gạch ngang
        tax_id = re.sub(r'[^\d\-]', '', tax_id)

        return tax_id

    def fit_transform(self, df, target_col=None):
        """
        Tiền xử lý dữ liệu: chuẩn hóa text, xử lý missing values, encoding.

        Args:
            df: DataFrame cần xử lý
            target_col: Tên cột target

        Returns:
            DataFrame đã được xử lý
        """
        logger.info(f"Bắt đầu tiền xử lý dữ liệu với shape: {df.shape}")

        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        processed_df = df.copy()

        # Lưu thông tin cột
        self.stats['columns'] = list(processed_df.columns)
        self.stats['dtypes'] = processed_df.dtypes.to_dict()
        self.stats['missing_values'] = processed_df.isnull().sum().to_dict()

        # Xác định các loại cột
        text_columns = processed_df.select_dtypes(include=['object']).columns.tolist()
        numeric_columns = processed_df.select_dtypes(include=np.number).columns.tolist()

        # Tìm cột mã số thuế (nếu có)
        tax_id_col = self._find_tax_id_column(processed_df)
        if tax_id_col:
            logger.info(f"Tìm thấy cột mã số thuế: {tax_id_col}")
            # Loại bỏ cột mã số thuế khỏi danh sách text columns nếu có
            if tax_id_col in text_columns:
                text_columns.remove(tax_id_col)

            # Chuẩn hóa cột mã số thuế
            processed_df[tax_id_col] = processed_df[tax_id_col].apply(self.normalize_tax_id)

        # Chuẩn hóa các cột text
        for col in text_columns:
            if col != target_col:  # Không chuẩn hóa cột target
                logger.info(f"Chuẩn hóa cột text: {col}")
                processed_df[col] = processed_df[col].apply(self.normalize_vietnamese_text)

        # Xử lý missing values
        # Cho cột số
        if numeric_columns:
            self.imputers['numeric'] = SimpleImputer(strategy='mean')
            processed_df[numeric_columns] = self.imputers['numeric'].fit_transform(
                processed_df[numeric_columns]
            )

        # Cho cột text
        if text_columns:
            self.imputers['text'] = SimpleImputer(strategy='most_frequent')
            processed_df[text_columns] = self.imputers['text'].fit_transform(
                processed_df[text_columns]
            )

        # Xử lý đặc biệt cho cột mã số thuế
        if tax_id_col:
            # Điền missing values cho cột mã số thuế bằng giá trị phổ biến nhất
            self.imputers['tax_id'] = SimpleImputer(strategy='most_frequent')
            processed_df[[tax_id_col]] = self.imputers['tax_id'].fit_transform(
                processed_df[[tax_id_col]]
            )

            # Tạo các features mới từ mã số thuế nếu cần
            processed_df = self._extract_tax_id_features(processed_df, tax_id_col)

        # Label encoding cho các cột categorical (trừ cột target)
        categorical_cols = [col for col in text_columns if col != target_col]
        if tax_id_col:
            categorical_cols.append(tax_id_col)

        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            processed_df[f"{col}_encoded"] = self.label_encoders[col].fit_transform(processed_df[col])

        # One-hot encoding nếu số lượng giá trị duy nhất ít
        for col in categorical_cols:
            unique_count = processed_df[col].nunique()
            if unique_count < 10:  # Ngưỡng cho one-hot encoding
                logger.info(f"Áp dụng one-hot encoding cho cột: {col} (có {unique_count} giá trị duy nhất)")
                dummies = pd.get_dummies(processed_df[col], prefix=col)
                processed_df = pd.concat([processed_df, dummies], axis=1)

        # Label encoding cho target nếu cần
        if target_col and target_col in processed_df.columns:
            self.label_encoders['target'] = LabelEncoder()
            processed_df[f"{target_col}_encoded"] = self.label_encoders['target'].fit_transform(
                processed_df[target_col])

            # Lưu mapping giữa target encoded và tên class
            self.target_mapping = {
                i: cls for i, cls in enumerate(self.label_encoders['target'].classes_)
            }
            logger.info(f"Target mapping: {self.target_mapping}")

        logger.info(f"Hoàn thành tiền xử lý dữ liệu. Shape mới: {processed_df.shape}")
        return processed_df

    def transform(self, df):
        """
        Áp dụng các bước tiền xử lý đã fit cho dữ liệu mới.

        Args:
            df: DataFrame cần xử lý

        Returns:
            DataFrame đã được xử lý
        """
        logger.info(f"Áp dụng tiền xử lý cho dữ liệu mới với shape: {df.shape}")

        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        processed_df = df.copy()

        # Xác định các loại cột
        text_columns = [col for col in self.label_encoders.keys()
                        if col != 'target' and col in df.columns]

        # Tìm cột mã số thuế (nếu đã xác định trước đó)
        tax_id_col = self._find_tax_id_column(processed_df)

        # Chuẩn hóa cột text
        for col in text_columns:
            if col == tax_id_col:
                processed_df[col] = processed_df[col].apply(self.normalize_tax_id)
            else:
                processed_df[col] = processed_df[col].apply(self.normalize_vietnamese_text)

        # Áp dụng imputers đã fit
        for imputer_type, imputer in self.imputers.items():
            if imputer_type == 'numeric':
                numeric_columns = processed_df.select_dtypes(include=np.number).columns.tolist()
                if numeric_columns:
                    processed_df[numeric_columns] = imputer.transform(processed_df[numeric_columns])
            elif imputer_type == 'text':
                if text_columns:
                    # Lọc các cột text có trong dataframe mới
                    cols_to_impute = [col for col in text_columns if col in processed_df.columns
                                      and col != tax_id_col]
                    if cols_to_impute:
                        processed_df[cols_to_impute] = imputer.transform(processed_df[cols_to_impute])
            elif imputer_type == 'tax_id' and tax_id_col:
                processed_df[[tax_id_col]] = imputer.transform(processed_df[[tax_id_col]])

        # Xử lý đặc biệt cho cột mã số thuế
        if tax_id_col and tax_id_col in processed_df.columns:
            processed_df = self._extract_tax_id_features(processed_df, tax_id_col)

        # Áp dụng label encoding
        for col, encoder in self.label_encoders.items():
            if col in processed_df.columns and col != 'target':
                # Xử lý các giá trị mới không có trong tập huấn luyện
                processed_df[col] = self._handle_unseen_categories(processed_df[col], encoder)
                processed_df[f"{col}_encoded"] = encoder.transform(processed_df[col])

        # One-hot encoding nếu đã được áp dụng trong fit_transform
        for col in text_columns:
            if col in processed_df.columns:
                unique_count = len(self.label_encoders.get(col).classes_)
                if unique_count < 10:  # Ngưỡng cho one-hot encoding
                    dummies = pd.get_dummies(processed_df[col], prefix=col)
                    # Đảm bảo tất cả các cột one-hot từ tập huấn luyện đều có mặt
                    for dummy_col in [f"{col}_{cls}" for cls in self.label_encoders[col].classes_]:
                        if dummy_col not in dummies.columns:
                            dummies[dummy_col] = 0
                    processed_df = pd.concat([processed_df, dummies], axis=1)

        logger.info(f"Hoàn thành áp dụng tiền xử lý. Shape mới: {processed_df.shape}")
        return processed_df

    def _find_tax_id_column(self, df):
        """
        Tìm cột có khả năng là mã số thuế dựa trên tên cột hoặc nội dung.
        """
        possible_names = ['ma_so_thue', 'mst', 'tax_id', 'tax_code', 'mã số thuế']

        # Tìm theo tên cột
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names):
                return col

        # Tìm theo mẫu dữ liệu - mã số thuế thường là 10-13 chữ số
        for col in df.select_dtypes(include=['object']).columns:
            # Kiểm tra 5 giá trị đầu tiên không null
            sample = df[col].dropna().head(5)
            if len(sample) > 0:
                tax_id_pattern = r'^\d{10,13}(-\d{3})?$'
                if all(re.match(tax_id_pattern, str(val)) for val in sample):
                    return col

        return None

    def _extract_tax_id_features(self, df, tax_id_col):
        """
        Trích xuất thông tin từ mã số thuế.
        Mã số thuế doanh nghiệp Việt Nam thường có một số đặc điểm như:
        - 10 chữ số cho doanh nghiệp thông thường
        - 13 chữ số (10-XXX) cho chi nhánh, đơn vị phụ thuộc
        """
        if tax_id_col not in df.columns:
            return df

        # Kiểm tra độ dài mã số thuế
        df[f'{tax_id_col}_length'] = df[tax_id_col].apply(lambda x: len(str(x)))

        # Kiểm tra có phải chi nhánh hay không (có dấu gạch ngang)
        df[f'{tax_id_col}_is_branch'] = df[tax_id_col].apply(lambda x: 1 if '-' in str(x) else 0)

        # Lấy 2 số đầu tiên của mã số thuế (thường cho biết tỉnh/thành phố)
        df[f'{tax_id_col}_province_code'] = df[tax_id_col].apply(
            lambda x: str(x)[:2] if pd.notna(x) and len(str(x)) >= 2 else "00"
        )

        return df

    def _handle_unseen_categories(self, series, encoder):
        """
        Xử lý các giá trị category mới không có trong tập huấn luyện.
        """
        # Lấy giá trị phổ biến nhất từ các categories đã biết
        known_categories = set(encoder.classes_)
        most_common = encoder.classes_[0]  # Giá trị đầu tiên làm mặc định

        # Thay thế các giá trị không có trong tập huấn luyện
        return series.apply(lambda x: x if x in known_categories else most_common)

    def save(self, path):
        """
        Lưu processor xuống file.
        """
        import joblib

        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Lưu processor
        joblib.dump(self, path)
        logger.info(f"Đã lưu processor vào {path}")

    @classmethod
    def load(cls, path):
        """
        Load processor từ file.
        """
        import joblib

        # Load processor
        processor = joblib.load(path)
        logger.info(f"Đã load processor từ {path}")

        return processor


def main():
    """
    Hàm main để chạy script từ command line.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Tiền xử lý dữ liệu cho mô hình phân loại')
    parser.add_argument('--input', required=True, help='Đường dẫn đến file dữ liệu đầu vào (CSV)')
    parser.add_argument('--output', required=True, help='Đường dẫn lưu file dữ liệu đã xử lý')
    parser.add_argument('--target', help='Tên cột target (nếu có)')
    parser.add_argument('--tax_id_col', help='Tên cột chứa mã số thuế (nếu biết trước)')
    parser.add_argument('--processor_path', help='Đường dẫn lưu/load processor')
    parser.add_argument('--mode', choices=['fit', 'transform'], default='fit',
                        help='Chế độ xử lý: fit (huấn luyện mới) hoặc transform (áp dụng đã có)')

    args = parser.parse_args()

    # Load dữ liệu
    logger.info(f"Đang đọc dữ liệu từ {args.input}")
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    elif args.input.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(args.input)
    else:
        raise ValueError("Chỉ hỗ trợ file CSV hoặc Excel")

    logger.info(f"Dữ liệu đầu vào có shape: {df.shape}")

    # Xử lý dữ liệu
    if args.mode == 'fit':
        # Tạo và fit processor mới
        processor = DataProcessor()
        processed_df = processor.fit_transform(df, target_col=args.target)

        # Lưu processor nếu cần
        if args.processor_path:
            processor.save(args.processor_path)
    else:
        # Load processor đã có
        if not args.processor_path:
            raise ValueError("Cần cung cấp đường dẫn processor khi sử dụng chế độ transform")

        processor = DataProcessor.load(args.processor_path)
        processed_df = processor.transform(df)

    # Lưu dữ liệu đã xử lý
    logger.info(f"Đang lưu dữ liệu đã xử lý vào {args.output}")
    if args.output.endswith('.csv'):
        processed_df.to_csv(args.output, index=False)
    elif args.output.endswith(('.xls', '.xlsx')):
        processed_df.to_excel(args.output, index=False)

    logger.info("Hoàn thành xử lý dữ liệu!")


if __name__ == "__main__":
    main()