import runpod
import traceback

try:
    import torch
    import io
    import os
    import numpy as np
    import zipfile
    
    # ¡CRÍTICO! Forzar la ruta del caché ANTES de importar cualquier inteligencia artificial
    # Si esto se pone después, Python lo ignora y descarga en el disco temporal de 5GB.
    os.environ["HF_HOME"] = "/runpod-volume/models/huggingface"
    
    import boto3
    import shutil
    import requests
    from PIL import Image
    from diffusers import QwenImageLayeredPipeline, QwenImageTransformer2DModel
    from diffusers.utils import load_image
    import onnxruntime as ort

    # Variables de entorno para Storage (Cloudflare R2)
    R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
    R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY")
    R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY")
    R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "pidfey-ephemeral")
    R2_PUBLIC_DOMAIN = os.environ.get("R2_PUBLIC_DOMAIN", "https://assets.pidfey.pro")

    # Rutas del Network Volume (El Rayo ⚡)
    QWEN_DIR = "/runpod-volume/models/Qwen-Image-Layered"
    QWEN_FILE = os.path.join(QWEN_DIR, "qwen_image_layered_fp8_e4m3fn.safetensors")
    ESRGAN_FILE = "/runpod-volume/models/RealESRGAN_x4plus_anime_6B.onnx"

    HF_TOKEN = os.environ.get("HF_TOKEN") # Opcional: Para evitar bloqueos de HuggingFace

    print("========= DIAGNÓSTICO DE DISCO DURO =========")
    import subprocess
    print("Espacio total y libre:")
    subprocess.run(["df", "-h", "/runpod-volume"], check=False)
    print("\nPeso de las carpetas dentro de /runpod-volume/models:")
    subprocess.run(["du", "-sh", "/runpod-volume/models/Qwen-Image-Layered"], check=False)
    subprocess.run(["du", "-sh", "/runpod-volume/models/huggingface"], check=False)
    print("===========================================")

    # Limpieza automática: Borrar el archivo FP8 defectuoso si aún existe para liberar 19 GB
    if os.path.exists(QWEN_FILE):
        try:
            print("Limpieza: Borrando archivo FP8 defectuoso para recuperar 19GB de espacio...")
            os.remove(QWEN_FILE)
            print("Archivo FP8 borrado con éxito.")
        except Exception as e:
            print(f"No se pudo borrar el FP8: {e}")

    # Limpieza automática: Extirpar SDXL del disco duro para recuperar 7 GB
    SDXL_DIR_TO_DELETE = "/runpod-volume/models/SDXL"
    if os.path.exists(SDXL_DIR_TO_DELETE):
        try:
            print(f"Limpieza: Borrando modelos pesados de SDXL en {SDXL_DIR_TO_DELETE}...")
            shutil.rmtree(SDXL_DIR_TO_DELETE)
            print("¡SDXL erradicado del disco duro exitosamente! (7GB liberados)")
        except Exception as e:
            print(f"Advertencia: No se pudo borrar SDXL: {e}")

    # ==========================================================
    # INSTALACIÓN AUTOMÁTICA DE REAL-CUGAN EN EL NETWORK VOLUME
    # ==========================================================
    CUGAN_BIN_DIR = "/runpod-volume/bin/cugan"
    CUGAN_EXE = os.path.join(CUGAN_BIN_DIR, "realcugan-ncnn-vulkan")
    
    if not os.path.exists(CUGAN_EXE):
        print("⚡ Instalando Real-CUGAN (Ubuntu) en el Network Volume por primera vez...")
        os.makedirs(CUGAN_BIN_DIR, exist_ok=True)
        # Descargamos la versión oficial de Ubuntu desde GitHub
        subprocess.run(["wget", "https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip", "-O", "/tmp/cugan.zip"], check=True)
        
        # Descomprimimos usando Python Nativo (evita errores de 'unzip' no instalado en RunPod)
        print("Descomprimiendo archivo ZIP de CUGAN...")
        with zipfile.ZipFile("/tmp/cugan.zip", 'r') as zip_ref:
            zip_ref.extractall("/tmp/")
            
        # Movemos los archivos al disco de red persistente
        subprocess.run(["mv", "/tmp/realcugan-ncnn-vulkan-20220728-ubuntu/realcugan-ncnn-vulkan", CUGAN_EXE], check=True)
        subprocess.run(["mv", "/tmp/realcugan-ncnn-vulkan-20220728-ubuntu/models-se", os.path.join(CUGAN_BIN_DIR, "models-se")], check=True)
        # Damos permisos de ejecución
        subprocess.run(["chmod", "+x", CUGAN_EXE], check=True)
        print("✅ Real-CUGAN (models-se) instalado exitosamente.")

    # ==========================================================
    # DEPENDENCIAS DEL SISTEMA (LIBRERÍAS C++ DE VULKAN)
    # ==========================================================
    if not os.path.exists("/usr/lib/x86_64-linux-gnu/libvulkan.so.1"):
        print("⚡ Instalando libvulkan1 (Dependencia crítica para CUGAN en RunPod)...")
        subprocess.run(["apt-get", "update"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["apt-get", "install", "-y", "libvulkan1"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ==========================================================
    # PRE-FLIGHT CHECK DE VULKAN (PREVENCIÓN DE CAÍDAS)
    # ==========================================================
    print("🛡️ Ejecutando diagnóstico de seguridad de Vulkan GPU...")
    VULKAN_READY = False
    try:
        # Creamos una imagen diminuta para prueba
        test_img = Image.new('RGB', (10, 10), color = 'black')
        test_in = "/tmp/test_vulkan_in.png"
        test_out = "/tmp/test_vulkan_out.png"
        test_img.save(test_in)
        
        # Ejecutamos CUGAN para ver si detecta la GPU o explota
        test_cmd = [CUGAN_EXE, "-i", test_in, "-o", test_out, "-s", "2"]
        test_result = subprocess.run(test_cmd, capture_output=True, text=True)
        
        # Si devuelve código 0 y el archivo sale, significa que la tarjeta gráfica (Vulkan) lo procesó.
        if test_result.returncode == 0 and os.path.exists(test_out):
            VULKAN_READY = True
            print("🚀 VULKAN ESTÁ ACTIVO Y ENLAZADO A LA GPU. Modo Súper Resolución ENABLED.")
        else:
            print("⚠️ ADVERTENCIA: RunPod tiene bloqueado el driver Vulkan en este contenedor.")
            print(f"Log de error: {test_result.stderr}")
            print("⚠️ Activando Plan B: Fallback automático a modo Lanczos (Seguridad máxima).")
    except Exception as e:
        print(f"⚠️ Error en Pre-flight Check: {e}. Activando Plan B (Lanczos).")

    print("Inicializando contenedor y cargando TUBERÍA DUAL en H100 (80GB VRAM)...")

    try:
        # 1. Cargar Qwen-Image-Layered (El Cirujano - VERSIÓN OFICIAL Y PURA)
        print("Cargando Qwen-Image-Layered (Versión Oficial BF16)...")
        pipeline_qwen = QwenImageLayeredPipeline.from_pretrained(
            "Qwen/Qwen-Image-Layered", 
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN
        ).to("cuda")
        pipeline_qwen.set_progress_bar_config(disable=True)
        
        print("¡Tubería Dual cargada exitosamente en VRAM!")
        
        # 2. Cargar Real-ESRGAN (El Músculo)
        print("Cargando Real-ESRGAN (ONNX)...")
        if os.path.exists(ESRGAN_FILE):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            esrgan_session = ort.InferenceSession(ESRGAN_FILE, providers=providers)
            print("¡Real-ESRGAN cargado exitosamente en VRAM!")
        else:
            print("⚠️ No se encontró Real-ESRGAN en el disco duro. Se omitirá el escalado.")
            esrgan_session = None

    except Exception as e:
        print(f"Advertencia Crítica: Fallo al cargar los modelos. Detalle: {e}")
        pipeline_qwen = None
        esrgan_session = None

    # Cliente S3 para Cloudflare R2
    s3_client = boto3.client(
        service_name ="s3",
        endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id = R2_ACCESS_KEY,
        aws_secret_access_key = R2_SECRET_KEY,
        region_name="auto",
    )

    def ai_upscale_layer(layer_img: Image.Image, scale: int = 4) -> Image.Image:
        """Súper Resolución Inteligente usando Real-CUGAN con Tiling."""
        if not VULKAN_READY:
            # Si Vulkan falló en el arranque, devolvemos la imagen original
            # El algoritmo de process_and_upload_layer se encargará de usar Lanczos
            return layer_img

        tmp_in = "/tmp/cugan_in.png"
        tmp_out = "/tmp/cugan_out.png"
        
        # 1. Guardar la imagen extraída de Qwen
        layer_img.save(tmp_in, "PNG")
        
        # 2. Ejecutar CUGAN con models-se y tiling 1024 (Optimizadísimo para H100)
        cmd = [
            CUGAN_EXE, 
            "-i", tmp_in, 
            "-o", tmp_out, 
            "-s", str(scale),
            "-m", os.path.join(CUGAN_BIN_DIR, "models-se"),
            "-n", "-1", # Ruido por defecto
            "-t", "1024" # Tiling optimizado para VRAM de gran capacidad
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(tmp_out):
            upscaled_img = Image.open(tmp_out).convert("RGBA")
            return upscaled_img
        else:
            print(f"Error en CUGAN durante Inferencia: {result.stderr}")
            return layer_img

    def process_and_upload_layer(layer_img: Image.Image, name_suffix: str, job_id: str, max_width_cm: int, max_height_cm: int, dpi: int):
        # 1. Calcular escalado proporcional inteligente (Aspect Ratio)
        orig_w, orig_h = layer_img.size
        # Tamaño máximo en píxeles
        max_target_w_px = int((max_width_cm / 2.54) * dpi)
        max_target_h_px = int((max_height_cm / 2.54) * dpi)
        
        # Calcular proporciones
        ratio_w = max_target_w_px / orig_w
        ratio_h = max_target_h_px / orig_h
        ratio = min(ratio_w, ratio_h) # Usamos el mínimo para que encaje sin salirse
        
        target_w = int(orig_w * ratio)
        target_h = int(orig_h * ratio)
        
        # 2. Redimensionar usando LANCZOS manteniendo proporción
        resized_img = layer_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # 3. Guardar con metadata DPI
        buffer = io.BytesIO()
        resized_img.save(buffer, format="PNG", dpi=(dpi, dpi))
        buffer.seek(0)
        
        # 4. Subir a R2
        file_key = f"jobs/{job_id}/{name_suffix}.png"
        s3_client.upload_fileobj(buffer, R2_BUCKET_NAME, file_key, ExtraArgs={"ContentType": "image/png"})
        return f"{R2_PUBLIC_DOMAIN}/{file_key}"

    def handler(job):
        job_input = job['input']
        job_id = job['id']
        
        # Parámetros enviados desde Vercel
        prompt = job_input.get('prompt', "A detailed 2D vector illustration, comic style")
        image_url = job_input.get('image_url')
        print_width_cm = int(job_input.get('print_width_cm', 28))
        print_height_cm = int(job_input.get('print_height_cm', 28))
        print_dpi = int(job_input.get('print_dpi', 300))
        strength = float(job_input.get('strength', 0.75)) # Qué tanto cambiar la imagen original
        
        if not pipeline_qwen:
            return {"error": "Los modelos no están cargados correctamente. Revisa los logs de inicialización."}

        if not image_url:
            return {"error": "Se requiere una image_url base para la transformación Img2Img."}

        print(f"Job {job_id}: Procesando a {print_width_cm}x{print_height_cm}cm ({print_dpi} DPI)")

        try:
            # Descargar imagen base
            response = requests.get(image_url)
            # IMPORTANTE: Convertimos a RGB plano para que CUGAN no choque con canales Alfa y genere bordes verdes
            input_image_rgb = Image.open(io.BytesIO(response.content)).convert("RGB") 
            
            # 1. ENGORDAR LA IMAGEN BASE CON REAL-CUGAN (El Músculo primero)
            # Esto evita los bordes verdes, porque segmentaremos la imagen ya estando en 4K.
            print(f"Job {job_id}: Aplicando Real-CUGAN a la imagen base plana...")
            upscaled_input = ai_upscale_layer(input_image_rgb, scale=4)
            
            # Inferencia Tubería Dual en la H100
            with torch.inference_mode():
                # Extraer capas con Qwen (El Cirujano en Alta Resolución)
                qwen_inputs = {
                    "image": upscaled_input,
                    "generator": torch.Generator(device='cuda').manual_seed(777),
                    "num_inference_steps": 30,
                    "layers": 4, 
                }
                qwen_output = pipeline_qwen(**qwen_inputs)
                output_image_layers = qwen_output.images[0]

            # PASO 3: Limpiar polvo, Upscale 4K con IA, Acoplar y Subir
            composite_img = None
            clean_layers = []
            
            for i, layer_img in enumerate(output_image_layers):
                # Limpiar polvo: Alpha < 20 = 0
                l_img = layer_img.convert("RGBA")
                r, g, b, a = l_img.split()
                alfa_np = np.array(a)
                alfa_np[alfa_np < 20] = 0
                alfa_limpio = Image.fromarray(alfa_np)
                clean_layer = Image.merge("RGBA", (r, g, b, alfa_limpio))
                
                clean_layers.append(clean_layer)
                
                # Ignoramos la capa 0 para el composite (fondo de Qwen)
                if i > 0:
                    if composite_img is None:
                        composite_img = clean_layer.copy()
                    else:
                        composite_img.alpha_composite(clean_layer)

            # Fallback en caso de que solo haya capa 0
            if composite_img is None:
                composite_img = clean_layers[0]

            layer_urls = []
            
            # Subir el composite (Acoplado)
            composite_url = process_and_upload_layer(composite_img, "final_composite_300dpi", job_id, print_width_cm, print_height_cm, print_dpi)
            
            # Subir capas segmentadas individuales
            for i, c_layer in enumerate(clean_layers):
                url = process_and_upload_layer(c_layer, f"layer_{i}", job_id, print_width_cm, print_height_cm, print_dpi)
                layer_urls.append({"name": f"layer_{i}.png", "url": url})
                
            return {
                "success": True,
                "message": "Extracción y Acoplado completado exitosamente.",
                "composite_url": composite_url,
                "layers": layer_urls
            }
            
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

    # Iniciar el Worker
    runpod.serverless.start({"handler": handler})

except Exception as boot_error:
    # SI FALLA EL BOOT (Importaciones, etc), LEVANTAMOS UN WORKER DUMMY PARA ATRAPAR EL ERROR
    error_traceback = traceback.format_exc()
    print("CRITICAL BOOT ERROR:")
    print(error_traceback)
    
    def crash_handler(job):
        return {
            "error": "CRITICAL BOOT ERROR", 
            "message": str(boot_error),
            "traceback": error_traceback
        }
        
    runpod.serverless.start({"handler": crash_handler})

