from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import json
import time
import os
import logging
import sys
from werkzeug.exceptions import RequestEntityTooLarge

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB máximo

# Variables globales
model = None
start_time = time.time()
request_count = 0
inference_count = 0


def setup_torch_hub():
    """Configurar torch.hub para evitar rate limits"""
    try:
        # Crear directorio de cache local
        cache_dir = './torch_hub_cache'
        os.makedirs(cache_dir, exist_ok=True)
        torch.hub.set_dir(cache_dir)

        # Configurar variables de entorno para GitHub
        os.environ['TORCH_HOME'] = cache_dir

        logger.info(f"Torch hub configurado con cache en: {cache_dir}")

    except Exception as e:
        logger.warning(f"No se pudo configurar torch.hub: {e}")


def load_model_alternative(model_path='impresion.pt'):
    """Método alternativo para cargar modelo YOLOv5 sin torch.hub"""
    global model

    if model is not None:
        return model

    try:
        logger.info("Intentando carga alternativa del modelo...")

        # Verificar si tenemos el código de YOLOv5 localmente
        yolo_path = './yolov5'  # Ruta donde estaría el código de YOLOv5

        if os.path.exists(yolo_path):
            logger.info("Usando YOLOv5 local...")
            sys.path.append(yolo_path)

            try:
                from models.experimental import attempt_load
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = attempt_load(model_path, map_location=device)
                model.eval()
                logger.info(f"Modelo cargado con YOLOv5 local en {device}")
                return model
            except ImportError:
                logger.warning("No se pudo importar YOLOv5 local")

        # Si no hay YOLOv5 local, usar torch.load directamente
        logger.info("Usando torch.load directo...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Cargar el checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Extraer el modelo del checkpoint
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model'].float()
            elif 'ema' in checkpoint:
                model = checkpoint['ema'].float()
            else:
                # Asumir que el checkpoint es el modelo completo
                model = checkpoint
        else:
            model = checkpoint

        # Configurar modelo
        model.to(device)
        model.eval()

        # Configurar nombres de clases si no existen
        if not hasattr(model, 'names'):
            model.names = {0: 'defect'}  # Nombre por defecto

        logger.info(f"Modelo cargado alternativamente en {device}")
        return model

    except Exception as e:
        logger.error(f"Error en carga alternativa: {e}")
        return None


def load_model_with_retry(model_path='impresion.pt', max_retries=3):
    """Carga el modelo YOLOv5 con manejo de rate limit - VERSIÓN MEJORADA"""
    global model

    if model is not None:
        logger.info("Modelo ya cargado, reutilizando...")
        return model

    # Verificar si el archivo del modelo existe
    if not os.path.exists(model_path):
        logger.error(f"Archivo del modelo no encontrado: {model_path}")
        raise FileNotFoundError(f"Archivo del modelo no encontrado: {model_path}")

    for attempt in range(max_retries):
        try:
            logger.info(f"Intento {attempt + 1} de cargar modelo...")

            # MÉTODO 1: Intentar cargar directamente con torch (sin torch.hub)
            try:
                logger.info("Intentando carga directa con torch.load()...")
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

                # Cargar modelo directamente
                checkpoint = torch.load(model_path, map_location=device)

                # Si el checkpoint contiene el modelo completo
                if 'model' in checkpoint:
                    model = checkpoint['model']
                else:
                    # Si es solo el estado del modelo
                    model = checkpoint

                model.to(device)
                model.eval()  # Modo evaluación

                logger.info(f"Modelo cargado directamente en {device}")
                return model

            except Exception as direct_error:
                logger.warning(f"Carga directa falló: {direct_error}")

                # MÉTODO 2: Usar torch.hub con configuración mejorada
                logger.info("Intentando con torch.hub con configuración anti-rate-limit...")

                # Configurar torch.hub para evitar rate limits
                torch.hub.set_dir('./torch_hub_cache')  # Usar cache local

                model = torch.hub.load(
                    'ultralytics/yolov5',
                    'custom',
                    path=model_path,
                    force_reload=False,  # No forzar descarga
                    trust_repo=True,
                    verbose=False,
                    skip_validation=True,  # Saltar validaciones innecesarias
                    source='local'  # Intentar usar fuente local primero
                )

                # Configurar dispositivo
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model.to(device)

                logger.info(f"Modelo cargado con torch.hub en {device}")
                return model

        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Error en intento {attempt + 1}: {str(e)}")

            # Detectar diferentes tipos de errores
            if any(keyword in error_msg for keyword in ["rate limit", "403", "forbidden", "too many requests"]):
                wait_time = min((2 ** attempt) * 30, 300)  # Max 5 minutos
                logger.info(f"Rate limit detectado. Esperando {wait_time} segundos...")
                time.sleep(wait_time)

            elif "connection" in error_msg or "timeout" in error_msg:
                wait_time = 30
                logger.info(f"Error de conexión. Esperando {wait_time} segundos...")
                time.sleep(wait_time)

            elif attempt == max_retries - 1:
                logger.error("Se agotaron todos los intentos de carga del modelo")
                raise e
            else:
                time.sleep(10)  # Espera corta para otros errores

    return None


def initialize_model():
    """Inicializa el modelo al arrancar la aplicación - VERSIÓN MEJORADA"""
    try:
        logger.info("Iniciando carga del modelo...")

        # Configurar torch.hub primero
        setup_torch_hub()

        # Intentar métodos en orden de preferencia
        methods = [
            ("Método directo", load_model_alternative),
            ("Método con reintentos", load_model_with_retry),
        ]

        for method_name, method_func in methods:
            try:
                logger.info(f"Probando {method_name}...")
                result = method_func()
                if result is not None:
                    logger.info(f"✅ Modelo cargado exitosamente con {method_name}")
                    return
            except Exception as e:
                logger.warning(f"{method_name} falló: {e}")
                continue

        logger.error("❌ Todos los métodos de carga fallaron")

    except Exception as e:
        logger.error(f"Error fatal al inicializar modelo: {e}")
        # En este caso, la app puede seguir funcionando y intentar cargar el modelo después


# Middleware para logging de requests
@app.before_request
def log_request_info():
    global request_count
    request_count += 1
    logger.info(f"Request #{request_count}: {request.method} {request.url}")
    if request.content_length:
        logger.info(f"Content-Length: {request.content_length} bytes")


@app.after_request
def log_response_info(response):
    logger.info(f"Response: {response.status_code}")
    return response


# Endpoints básicos
@app.route("/", methods=["GET"])
def root():
    """Endpoint raíz con información de la API"""
    global model, request_count, inference_count

    return jsonify({
        "message": "API de Inferencia YOLOv5 - Versión Anti-Rate-Limit",
        "status": "active",
        "model_loaded": model is not None,
        "uptime_seconds": round(time.time() - start_time, 2),
        "total_requests": request_count,
        "total_inferences": inference_count,
        "endpoints": {
            "POST /inferencia": "Realizar inferencia en imagen",
            "GET /health": "Verificar estado detallado de la API",
            "GET /ping": "Ping simple para keep-alive",
            "GET /stats": "Estadísticas del servidor",
            "POST /model/reload": "Recargar modelo manualmente"
        },
        "supported_formats": ["png", "jpg", "jpeg", "gif", "bmp", "tiff"],
        "max_file_size": "16MB",
        "version": "2.1-anti-rate-limit"
    })


@app.route("/ping", methods=["GET"])
def ping():
    """Endpoint simple para mantener el servidor despierto"""
    return jsonify({
        "status": "pong",
        "timestamp": time.time(),
        "uptime": round(time.time() - start_time, 2),
        "model_loaded": model is not None
    })


@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint mejorado para verificar estado de la API"""
    global model, request_count, inference_count

    health_data = {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "timestamp": time.time(),
        "uptime_seconds": round(time.time() - start_time, 2),
        "total_requests": request_count,
        "total_inferences": inference_count,
        "memory_info": "Available" if model is not None else "Model not loaded",
        "torch_hub_cache": os.path.exists('./torch_hub_cache')
    }

    if model is not None:
        try:
            health_data["device"] = str(next(model.parameters()).device)
            health_data["model_ready"] = True
            health_data["model_type"] = "YOLOv5 Custom"
        except Exception as e:
            health_data["model_ready"] = False
            health_data["model_error"] = str(e)
    else:
        health_data["device"] = "unknown"
        health_data["model_ready"] = False

    # Determinar código de estado HTTP
    status_code = 200 if model is not None else 503

    return jsonify(health_data), status_code


@app.route("/stats", methods=["GET"])
def get_stats():
    """Obtener estadísticas detalladas del servidor"""
    global model, request_count, inference_count

    uptime_seconds = time.time() - start_time

    stats = {
        "server_info": {
            "uptime_seconds": round(uptime_seconds, 2),
            "uptime_formatted": format_uptime(uptime_seconds),
            "start_time": start_time,
            "current_time": time.time(),
            "version": "2.1-anti-rate-limit"
        },
        "request_stats": {
            "total_requests": request_count,
            "total_inferences": inference_count,
            "avg_requests_per_minute": round((request_count / (uptime_seconds / 60)) if uptime_seconds > 0 else 0, 2)
        },
        "model_info": {
            "loaded": model is not None,
            "device": str(next(model.parameters()).device) if model is not None else "unknown",
            "model_file": "impresion.pt"
        },
        "system_info": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "torch_hub_cache_exists": os.path.exists('./torch_hub_cache')
        }
    }

    return jsonify(stats)


def format_uptime(seconds):
    """Formatear tiempo de actividad de forma legible"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"


@app.route("/inferencia", methods=["POST"])
def inferencia():
    """Endpoint de inferencia mejorado con manejo robusto de modelos"""
    global model, inference_count
    request_start = time.time()

    try:
        logger.info("Nueva solicitud de inferencia recibida")

        # Verificar que el modelo esté cargado
        if model is None:
            logger.warning("Modelo no cargado, intentando cargar...")

            # Intentar cargar con múltiples métodos
            try:
                load_model_alternative()
                if model is None:
                    load_model_with_retry()
            except Exception as load_error:
                logger.error(f"Error cargando modelo: {load_error}")

            if model is None:
                logger.error("No se pudo cargar el modelo")
                return jsonify({
                    "error": "Modelo no disponible",
                    "message": "El modelo está cargándose o hay problemas de conectividad. Intente nuevamente en unos minutos.",
                    "status": "model_loading",
                    "suggestion": "El servidor puede estar experimentando rate limits. Reintente en 1-2 minutos."
                }), 503

        # Verificar que se envió un archivo
        if "file" not in request.files:
            logger.warning("No se envió archivo en la solicitud")
            return jsonify({"error": "No se envió ninguna imagen"}), 400

        image_file = request.files["file"]
        file_size = request.content_length or 0
        logger.info(f"Archivo recibido: {image_file.filename}, tamaño: {file_size} bytes")

        # Verificar que el archivo no esté vacío
        if image_file.filename == '':
            return jsonify({"error": "No se seleccionó ningún archivo"}), 400

        # Verificar tipo de archivo
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        if not ('.' in image_file.filename and
                image_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({
                "error": "Tipo de archivo no permitido",
                "allowed_formats": list(allowed_extensions)
            }), 400

        # Cargar y procesar imagen
        try:
            img_load_start = time.time()
            img = Image.open(image_file.stream)

            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_load_time = time.time() - img_load_start
            logger.info(f"Imagen cargada en {img_load_time:.3f}s, tamaño: {img.size}, modo: {img.mode}")

        except Exception as e:
            logger.error(f"Error procesando imagen: {str(e)}")
            return jsonify({"error": f"Error al procesar imagen: {str(e)}"}), 400

        # Realizar inferencia
        try:
            inference_start = time.time()
            logger.info("Iniciando inferencia...")

            results = model(img)
            inference_time = time.time() - inference_start
            inference_count += 1

            # Convertir resultados a formato JSON más limpio
            detections = results.pandas().xyxy[0]

            # Crear respuesta estructurada
            response_data = {
                "success": True,
                "detections": json.loads(detections.to_json(orient="records")),
                "detection_count": len(detections),
                "image_info": {
                    "filename": image_file.filename,
                    "size": {
                        "width": img.width,
                        "height": img.height
                    },
                    "mode": img.mode,
                    "file_size_bytes": file_size
                },
                "model_info": {
                    "device": str(next(model.parameters()).device),
                    "model_name": "YOLOv5 Custom",
                    "model_file": "impresion.pt",
                    "version": "2.1-anti-rate-limit"
                },
                "processing_time": {
                    "image_load_seconds": round(img_load_time, 3),
                    "inference_seconds": round(inference_time, 3),
                    "total_seconds": round(time.time() - request_start, 3)
                },
                "timestamp": time.time(),
                "inference_id": inference_count
            }

            logger.info(
                f"Inferencia #{inference_count} completada en {inference_time:.3f}s. Detecciones: {len(detections)}")

            # Log detallado de detecciones si las hay
            if len(detections) > 0:
                for idx, detection in enumerate(detections.itertuples()):
                    logger.info(f"  Detección {idx + 1}: {detection.name} (confianza: {detection.confidence:.3f})")

            return jsonify(response_data)

        except Exception as e:
            logger.error(f"Error durante inferencia: {str(e)}")
            return jsonify({
                "error": f"Error durante inferencia: {str(e)}",
                "processing_time": {
                    "total_seconds": round(time.time() - request_start, 3)
                }
            }), 500

    except RequestEntityTooLarge:
        logger.warning("Archivo demasiado grande recibido")
        return jsonify({
            "error": "Archivo demasiado grande (máximo 16MB)",
            "max_size_mb": 16
        }), 413
    except Exception as e:
        logger.error(f"Error general: {str(e)}")
        return jsonify({
            "error": f"Error interno del servidor: {str(e)}",
            "processing_time": {
                "total_seconds": round(time.time() - request_start, 3)
            }
        }), 500
    finally:
        total_time = time.time() - request_start
        logger.info(f"Solicitud de inferencia completada en {total_time:.3f}s")


@app.route("/model/reload", methods=["POST"])
def reload_model():
    """Endpoint para recargar el modelo manualmente con métodos mejorados"""
    global model
    try:
        logger.info("Recarga manual del modelo solicitada")
        model = None  # Reset del modelo

        # Intentar múltiples métodos de carga
        success = False
        methods = [
            ("Método alternativo", load_model_alternative),
            ("Método con reintentos", load_model_with_retry)
        ]

        for method_name, method_func in methods:
            try:
                logger.info(f"Intentando recarga con {method_name}...")
                result = method_func()
                if result is not None:
                    logger.info(f"✅ Modelo recargado exitosamente con {method_name}")
                    success = True
                    break
            except Exception as e:
                logger.warning(f"{method_name} falló: {e}")
                continue

        if success and model is not None:
            return jsonify({
                "message": "Modelo recargado exitosamente",
                "device": str(next(model.parameters()).device),
                "timestamp": time.time(),
                "version": "2.1-anti-rate-limit"
            })
        else:
            logger.error("Error al recargar modelo con todos los métodos")
            return jsonify({
                "error": "Error al recargar modelo",
                "message": "Todos los métodos de carga fallaron"
            }), 500

    except Exception as e:
        logger.error(f"Error al recargar modelo: {str(e)}")
        return jsonify({"error": f"Error al recargar modelo: {str(e)}"}), 500


@app.route("/model/info", methods=["GET"])
def model_info():
    """Obtener información detallada del modelo"""
    global model

    if model is None:
        return jsonify({
            "model_loaded": False,
            "error": "Modelo no cargado",
            "suggestion": "Use POST /model/reload para intentar cargar el modelo"
        }), 503

    try:
        model_info_data = {
            "model_loaded": True,
            "device": str(next(model.parameters()).device),
            "model_type": "YOLOv5 Custom",
            "model_file": "impresion.pt",
            "classes": list(model.names.values()) if hasattr(model, 'names') else "Unknown",
            "num_classes": len(model.names) if hasattr(model, 'names') else "Unknown",
            "input_size": "Variable (YOLOv5 auto-resize)",
            "framework": "PyTorch",
            "torch_version": torch.__version__,
            "version": "2.1-anti-rate-limit"
        }

        return jsonify(model_info_data)

    except Exception as e:
        return jsonify({
            "model_loaded": True,
            "error": f"Error obteniendo información del modelo: {str(e)}"
        }), 500


# Manejadores de errores
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint no encontrado",
        "available_endpoints": [
            "GET /",
            "GET /health",
            "GET /ping",
            "GET /stats",
            "GET /model/info",
            "POST /inferencia",
            "POST /model/reload"
        ]
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Método no permitido",
        "message": "Verifique el método HTTP usado para este endpoint"
    }), 405


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Error interno del servidor: {str(error)}")
    return jsonify({
        "error": "Error interno del servidor",
        "message": "Por favor contacte al administrador si el problema persiste"
    }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": "Archivo demasiado grande",
        "max_size_mb": 16,
        "message": "El archivo excede el tamaño máximo permitido"
    }), 413


# Inicializar modelo al importar el módulo (para Flask 2.2+)
with app.app_context():
    initialize_model()

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 INICIANDO API DE INFERENCIA YOLOV5 - VERSIÓN ANTI-RATE-LIMIT")
    logger.info("=" * 60)
    logger.info(f"🐍 Versión de Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    logger.info(f"🔥 PyTorch versión: {torch.__version__}")
    logger.info(f"🎯 CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"🎮 Dispositivos CUDA: {torch.cuda.device_count()}")
    logger.info(f"📁 Archivo de modelo: impresion.pt")
    logger.info(f"🌐 Puerto: 5000")
    logger.info(f"🔧 Tamaño máximo de archivo: 16MB")
    logger.info(f"📦 Cache torch.hub: ./torch_hub_cache")
    logger.info("=" * 60)

    # Ejecutar la aplicación
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,  # Cambiar a False en producción
        threaded=True
    )