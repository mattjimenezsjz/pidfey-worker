import runpod
import traceback

try:
    import torch
    import io
    import os
    import numpy as np
    import subprocess
    import concurrent.futures
    
    # ¡CRÍTICO! Forzar la ruta del caché ANTES de importar cualquier inteligencia artificial
    # Si esto se pone después, Python lo ignora y descarga en el disco temporal de 5GB.
    os.environ["HF_HOME"] = "/runpod-volume/models/huggingface"
    
    # (Los upscalers CUGAN y Real-ESRGAN han sido removidos para preservar la resolución nativa y semitonos)
    
    import boto3
    import shutil
    import requests
    from PIL import Image
    from diffusers import QwenImageLayeredPipeline, QwenImageTransformer2DModel
    from diffusers.utils import load_image

    # Variables de entorno para Storage (Cloudflare R2)
    R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
    R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY")
    R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY")
    R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "pidfey-ephemeral")
    R2_PUBLIC_DOMAIN = os.environ.get("R2_PUBLIC_DOMAIN", "https://assets.pidfey.pro")

    # Rutas del Network Volume (El Rayo ⚡)
    QWEN_DIR = "/runpod-volume/models/Qwen-Image-Layered"
    QWEN_FILE = os.path.join(QWEN_DIR, "qwen_image_layered_fp8_e4m3fn.safetensors")

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
    except Exception as e:
        print(f"Advertencia Crítica: Fallo al cargar los modelos. Detalle: {e}")
        pipeline_qwen = None

    # Cliente S3 para Cloudflare R2
    s3_client = boto3.client(
        service_name ="s3",
        endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id = R2_ACCESS_KEY,
        aws_secret_access_key = R2_SECRET_KEY,
        region_name="auto",
    )

    def process_and_upload_layer(layer_img: Image.Image, name_suffix: str, job_id: str, max_width_cm: int, max_height_cm: int):
        # 1. Guardar la imagen en su resolución original nativa 2K sin interpolación
        buffer = io.BytesIO()
        layer_img.save(buffer, format="PNG", compress_level=1)
        buffer.seek(0)
        
        # 4. Subir a R2
        file_key = f"jobs/{job_id}/{name_suffix}.png"
        s3_client.upload_fileobj(buffer, R2_BUCKET_NAME, file_key, ExtraArgs={"ContentType": "image/png"})
        return f"{R2_PUBLIC_DOMAIN}/{file_key}"

    def process_layer_with_math(layer_img: Image.Image, job_id: str, suffix: str, preset: str):
        # Aseguramos RGBA y pasamos a numpy float32
        layer_img = layer_img.convert("RGBA")
        img_np = np.array(layer_img).astype(np.float32)
        
        # Extraemos el canal alpha
        a = img_np[:, :, 3]
        
        # Limpiar polvo: Alpha < 20 = 0 (Conservamos los semitonos de Qwen intactos)
        a[a < 20] = 0
        
        # Actualizamos el canal alpha
        img_np[:, :, 3] = a
        
        # Retornamos la imagen limpia en su resolución nativa
        return Image.fromarray(img_np.astype(np.uint8), "RGBA")

    def handler(job):
        job_input = job['input']
        job_id = job['id']
        
        prompt = job_input.get('prompt', "A detailed 2D vector illustration, comic style")
        preset = job_input.get('preset', 'lineart')
        image_url = job_input.get('image_url')
        print_width_cm = int(job_input.get('print_width_cm', 28))
        print_height_cm = int(job_input.get('print_height_cm', 28))
        strength = float(job_input.get('strength', 0.75))
        
        if not pipeline_qwen:
            return {"error": "Los modelos no están cargados correctamente. Revisa los logs de inicialización."}

        if not image_url:
            return {"error": "Se requiere una image_url base para la transformación Img2Img."}

        print(f"Job {job_id}: Procesando a {print_width_cm}x{print_height_cm}cm")
        
        try:
            # 1. Descargar imagen base
            response = requests.get(image_url)
            input_image = Image.open(io.BytesIO(response.content)).convert("RGBA")
            
            # 2. Inferencia Qwen Nativa en H100, forzando la actualizacion.
            with torch.inference_mode():
                qwen_inputs = {
                    "image": input_image,
                    "generator": torch.Generator(device='cuda').manual_seed(777),
                    "num_inference_steps": 35,
                    "layers": 4,
                    "use_en_prompt": True,
                    
                }
                qwen_output = pipeline_qwen(**qwen_inputs)
                output_image_layers = qwen_output.images[0]

            # 3. Limpieza de Polvo en Paralelo (Multihilo)
            print("Lanzando filtro de limpieza de polvo en paralelo...")
            clean_layers = [None] * len(output_image_layers)
            layer_urls = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(output_image_layers)) as executor:
                futuros = {}
                for i, layer_img in enumerate(output_image_layers):
                    futuro = executor.submit(process_layer_with_math, layer_img, job_id, f"layer_{i}", preset)
                    futuros[futuro] = i
                    
                for futuro in concurrent.futures.as_completed(futuros):
                    indice = futuros[futuro]
                    try:
                        clean_layers[indice] = futuro.result()
                        print(f"Capa {indice} limpiada exitosamente.")
                    except Exception as e:
                        print(f"Error limpiando la capa {indice}: {e}")
                        raise e
                        
            # Ensamblamos el composite respetando el orden de las capas
            composite_img = None
            for i, upscaled_layer in enumerate(clean_layers):
                if upscaled_layer is None: continue
                # Ignoramos la capa 0 (fondo) para el composite final
                if i > 0:
                    if composite_img is None:
                        composite_img = upscaled_layer.copy()
                    else:
                        composite_img.alpha_composite(upscaled_layer)

            # Fallback
            if composite_img is None and len(clean_layers) > 0:
                composite_img = clean_layers[0]
            
            # 4. Subir a R2 en Paralelo (Multihilo)
            print("Iniciando subidas simultáneas a Cloudflare R2...")
            
            # Preparamos las tareas de subida
            upload_tasks = []
            
            # Tarea para el composite
            upload_tasks.append(
                ("final_composite", composite_img)
            )
            
            # Tareas para las capas individuales
            for i, c_layer in enumerate(clean_layers):
                if c_layer is not None:
                    upload_tasks.append((f"layer_{i}", c_layer))
                    
            # Ejecutamos las subidas en paralelo
            upload_results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(upload_tasks)) as executor:
                futuros_upload = {
                    executor.submit(process_and_upload_layer, task_img, task_name, job_id, print_width_cm, print_height_cm): task_name
                    for task_name, task_img in upload_tasks
                }
                for futuro in concurrent.futures.as_completed(futuros_upload):
                    t_name = futuros_upload[futuro]
                    try:
                        upload_results[t_name] = futuro.result()
                    except Exception as e:
                        print(f"Error subiendo {t_name} a R2: {e}")
                        
            composite_url = upload_results.get("final_composite", "")
            for i in range(len(clean_layers)):
                name = f"layer_{i}"
                if name in upload_results:
                    layer_urls.append({"name": f"{name}.png", "url": upload_results[name]})
                
            return {
                "success": True,
                "message": "Extracción Nativa, Limpieza de Polvo y Subida Paralela completados.",
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
