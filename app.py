from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import json
import time
import os
import logging
from werkzeug.exceptions import RequestEntityTooLarge

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB máximo

# Variable global para el modelo
model = None


def load_model_with_retry(model_path='impresion.pt', max_retries=3):
    """Carga el modelo YOLOv5 con manejo de rate limit"""
    global model

    if model is not None:
        logger.info("Modelo ya cargado, reutilizando...")
        return model

    for attempt in range(max_retries):
        try:
            logger.info(f"Intento {attempt + 1} de cargar modelo...")

            # Verificar si el archivo del modelo existe
            if not os.path.exists(model_path):
                logger.error(f"Archivo del modelo no encontrado: {model_path}")
                raise FileNotFoundError(f"Archivo del modelo no encontrado: {model_path}")

            # Cargar modelo sin force_reload para evitar descargas innecesarias
            model = torch.hub.load(
                'ultralytics/yolov5',
                'custom',
                path=model_path,
                force_reload=False,  # Cambio importante: evita recargas innecesarias
                trust_repo=True,
                verbose=False
            )

            # Configurar dispositivo (CPU/GPU)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)

            logger.info(f"Modelo cargado exitosamente en {device}")
            return model

        except Exception as e:
            logger.error(f"Error en intento {attempt + 1}: {str(e)}")

            if "rate limit" in str(e).lower() or "403" in str(e):
                wait_time = (2 ** attempt) * 30  # Backoff exponencial: 30s, 60s, 120s
                logger.info(f"Rate limit detectado. Esperando {wait_time} segundos...")
                time.sleep(wait_time)
            elif attempt == max_retries - 1:
                logger.error("Se agotaron todos los intentos de carga del modelo")
                raise e
            else:
                time.sleep(10)  # Espera corta para otros errores

    return None


def initialize_model():
    """Inicializa el modelo al arrancar la aplicación"""
    try:
        load_model_with_retry()
        logger.info("Inicialización del modelo completada")
    except Exception as e:
        logger.error(f"Error fatal al inicializar modelo: {e}")
        # En producción, podrías decidir no arrancar la app sin modelo
        pass


@app.before_first_request
def startup():
    """Se ejecuta antes de la primera solicitud"""
    initialize_model()


@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint para verificar estado de la API"""
    global model
    status = "healthy" if model is not None else "model_not_loaded"
    return jsonify({
        "status": status,
        "model_loaded": model is not None,
        "device": str(next(model.parameters()).device) if model is not None else "unknown"
    })


@app.route("/inferencia", methods=["POST"])
def inferencia():
    global model

    try:
        # Verificar que el modelo esté cargado
        if model is None:
            logger.warning("Modelo no cargado, intentando cargar...")
            load_model_with_retry()
            if model is None:
                return jsonify({"error": "Modelo no disponible"}), 503

        # Verificar que se envió un archivo
        if "file" not in request.files:
            return jsonify({"error": "No se envió ninguna imagen"}), 400

        image_file = request.files["file"]

        # Verificar que el archivo no esté vacío
        if image_file.filename == '':
            return jsonify({"error": "No se seleccionó ningún archivo"}), 400

        # Verificar tipo de archivo
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        if not ('.' in image_file.filename and
                image_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({"error": "Tipo de archivo no permitido"}), 400

        # Cargar y procesar imagen
        try:
            img = Image.open(image_file.stream)

            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')

        except Exception as e:
            return jsonify({"error": f"Error al procesar imagen: {str(e)}"}), 400

        # Realizar inferencia
        try:
            logger.info("Realizando inferencia...")
            results = model(img)

            # Convertir resultados a formato JSON más limpio
            detections = results.pandas().xyxy[0]

            # Crear respuesta estructurada
            response_data = {
                "detections": json.loads(detections.to_json(orient="records")),
                "detection_count": len(detections),
                "image_size": {
                    "width": img.width,
                    "height": img.height
                },
                "model_info": {
                    "device": str(next(model.parameters()).device),
                    "model_name": "YOLOv5 Custom"
                }
            }

            logger.info(f"Inferencia completada. Detecciones: {len(detections)}")
            return jsonify(response_data)

        except Exception as e:
            logger.error(f"Error durante inferencia: {str(e)}")
            return jsonify({"error": f"Error durante inferencia: {str(e)}"}), 500

    except RequestEntityTooLarge:
        return jsonify({"error": "Archivo demasiado grande (máximo 16MB)"}), 413
    except Exception as e:
        logger.error(f"Error general: {str(e)}")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500


@app.route("/model/reload", methods=["POST"])
def reload_model():
    """Endpoint para recargar el modelo manualmente"""
    global model
    try:
        model = None  # Reset del modelo
        load_model_with_retry()
        return jsonify({"message": "Modelo recargado exitosamente"})
    except Exception as e:
        return jsonify({"error": f"Error al recargar modelo: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def root():
    """Endpoint raíz con información de la API"""
    return jsonify({
        "message": "API de Inferencia YOLOv5",
        "endpoints": {
            "POST /inferencia": "Realizar inferencia en imagen",
            "GET /health": "Verificar estado de la API",
            "POST /model/reload": "Recargar modelo manualmente"
        },
        "supported_formats": ["png", "jpg", "jpeg", "gif", "bmp", "tiff"],
        "max_file_size": "16MB"
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint no encontrado"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Error interno del servidor"}), 500


if __name__ == "__main__":
    # Inicializar modelo al arrancar
    initialize_model()

    # Ejecutar la aplicación
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,  # Cambiar a False en producción
        threaded=True
    )