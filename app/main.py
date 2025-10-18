from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import asyncio

from app.core.config import settings
from app.core.async_predictor import AsyncMLPredictor
from app.utils.logger import get_logger
from app.core.dependencies import set_dependencies

logger = get_logger(__name__)

# Глобальные переменные
predictor: AsyncMLPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager для управления жизненным циклом приложения"""
    # Startup
    global predictor
    startup_time = time.time()

    logger.info("🚀 Запускаем Price Optimizer API...")

    try:
        # Проверяем существование модели
        if not os.path.exists(settings.MODEL_PATH):
            logger.error(f"❌ Файл модели не найден: {settings.MODEL_PATH}")
            raise FileNotFoundError(f"Модель не найдена: {settings.MODEL_PATH}")

        logger.info(f"✅ Модель найдена: {settings.MODEL_PATH}")

        # Инициализируем predictor
        predictor = AsyncMLPredictor(settings.MODEL_PATH)
        logger.info("🔄 Инициализируем ML predictor...")

        await predictor.initialize()
        logger.info("✅ ML predictor инициализирован!")

        # Устанавливаем зависимости
        set_dependencies(predictor, startup_time)
        logger.info("✅ Зависимости установлены!")

        logger.info("🎉 Приложение успешно запущено!")
        yield

    except Exception as e:
        logger.error(f"❌ Ошибка запуска приложения: {e}", exc_info=True)
        predictor = None
        raise

    finally:
        # Shutdown
        logger.info("🛑 Останавливаем приложение...")
        if predictor:
            await predictor.shutdown()
        logger.info("✅ Приложение остановлено")


# Создаем приложение
app = FastAPI(
    title="Price Optimizer API",
    version="1.0.0",
    description="API для оптимизации цен такси с ML",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Импортируем роутеры
from app.api.endpoints import router as api_router

app.include_router(api_router, prefix="/api/v1", tags=["API"])

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Основные эндпоинты
@app.get("/")
async def root():
    return {
        "message": "Welcome to Price Optimizer API!",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "get_optimal_prices": "POST /api/v1/get_optimal_prices",
            "health": "GET /api/v1/health",
            "stats": "GET /api/v1/stats"
        }
    }


@app.get("/ready")
async def readiness_check():
    if predictor and predictor.model_loaded:
        return {"status": "ready"}
    else:
        raise HTTPException(
            status_code=503,
            detail="Service not ready: model not loaded"
        )


@app.get("/live")
async def liveness_check():
    return {"status": "alive"}


# Обработчики ошибок
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"❌ Необработанная ошибка: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,
        log_level="info",
        access_log=settings.DEBUG
    )