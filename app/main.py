from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import asyncio
import platform
import signal

from app.core.config import settings
from app.core.async_predictor import AsyncMLPredictor
from app.utils.logger import get_logger
from app.core.dependencies import set_dependencies

logger = get_logger(__name__)

# Глобальные переменные
predictor: AsyncMLPredictor = None


def handle_shutdown(sig=None, frame=None):
    """Универсальный обработчик shutdown"""
    logger.info(f"🛑 Received shutdown signal {sig}, stopping application...")


def setup_signal_handlers():
    """Кросс-платформенная настройка сигналов"""
    if platform.system() != "Windows":
        # ✅ LINUX SIGNAL HANDLERS
        signal.signal(signal.SIGTERM, handle_shutdown)  # Kubernetes/Docker stop
        signal.signal(signal.SIGINT, handle_shutdown)  # Ctrl+C
        signal.signal(signal.SIGHUP, handle_shutdown)  # Terminal closed
        logger.info("✅ Linux signal handlers установлены")
    else:
        # ✅ WINDOWS SIGNAL HANDLERS
        try:
            import win32api
            win32api.SetConsoleCtrlHandler(handle_shutdown, True)
            logger.info("✅ Windows signal handlers установлены")
        except ImportError:
            logger.warning("⚠️ win32api not available, using default signal handling")


# Вызов настройки сигналов при импорте
setup_signal_handlers()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager для управления жизненным циклом приложения"""
    # Startup
    global predictor
    startup_time = time.time()

    logger.info(f"🚀 Запускаем Price Optimizer API на {platform.system()}...")

    try:
        # ✅ КРОССПЛАТФОРМЕННАЯ ПРОВЕРКА МОДЕЛИ
        model_path = settings.model_path

        if not os.path.exists(model_path):
            # Для Linux пробуем альтернативные пути
            if platform.system() != "Windows":
                alternative_paths = [
                    "models/catboost_taxi_smart.joblib",
                    "./catboost_taxi_smart.joblib"
                ]
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        logger.info(f"✅ Найдена модель по альтернативному пути: {alt_path}")
                        break
                else:
                    raise FileNotFoundError(f"Модель не найдена. Проверенные пути: {alternative_paths}")
            else:
                raise FileNotFoundError(f"Модель не найдена: {model_path}")

        logger.info(f"✅ Модель найдена: {model_path}")

        # Инициализируем predictor
        predictor = AsyncMLPredictor(model_path)
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
    allow_origins=["*"] if settings.DEBUG else [
        "https://your-production-domain.com",
        "https://admin.your-domain.com"
    ],
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
        "platform": platform.system(),
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
    # ✅ КРОССПЛАТФОРМЕННЫЙ ЗАПУСК
    uvicorn_config = {
        "app": "app.main:app",
        "host": settings.HOST,
        "port": settings.PORT,
        "reload": settings.DEBUG,
        "workers": 1,
        "log_level": "info",
        "access_log": settings.DEBUG
    }

    # Оптимизации для Linux
    if platform.system() != "Windows":
        uvicorn_config.update({
            "timeout_keep_alive": 5,
            "timeout_notify": 30,
            "timeout_graceful_shutdown": 30
        })

    uvicorn.run(**uvicorn_config)