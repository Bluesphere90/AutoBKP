FROM python:3.10-slim

WORKDIR /app

# Sao chép requirements trước
COPY ../requirements.txt .

# Cài đặt các phụ thuộc
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn
COPY .. .

# Tạo các thư mục cần thiết
RUN mkdir -p /app/data/models /app/data/processed /app/logs

# Cài đặt package
RUN pip install -e .

# Expose cổng
EXPOSE 8000

# Chạy API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]