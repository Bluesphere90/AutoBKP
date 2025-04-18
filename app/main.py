import uvicorn
from fastapi import FastAPI
from app.core.config import settings
from app.api.routes import router as api_router

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    description="Multi-class Classification API",
    version="0.1.0",
)

# Thêm router API
app.include_router(api_router, prefix=settings.API_PREFIX)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )