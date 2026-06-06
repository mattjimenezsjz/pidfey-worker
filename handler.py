import runpod
import traceback

try:
    import torch
    import io
    import os
    import boto3
    import requests
    from PIL import Image
    from diffusers import QwenImageLayeredPipeline, AutoPipelineForImage2Image, QwenImageTransformer2DModel
    from diffusers.utils import load_image

    # Variables de entorno para Storage (Cloudflare R2)
    R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
    R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY")
    R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY")
    R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "pidfey-ephemeral")
    R2_PUBLIC_DOMAIN = os.environ.get("R2_PUBLIC_DOMAIN", "https://assets.pidfey.pro")

    # Rutas del Network Volume (El Rayo ⚡)
    SDXL_DIR = "/runpod-volume/models/SDXL"
    QWEN_DIR = "/runpod-volume/models/Qwen-Image-Layered"
    QWEN_FILE = os.path.join(QWEN_DIR, "qwen_image_layered_fp8_e4m3fn.safetensors")

    # Configurar HuggingFace cache en el Network Volume para no descargar 2 veces
    os.environ["HF_HOME"] = "/runpod-volume/models/huggingface"
    HF_TOKEN = os.environ.get("HF_TOKEN") # Opcional: Para evitar bloqueos de HuggingFace

    print("========= DIAGNÓSTICO DE DISCO DURO =========")
    import subprocess
    print("Espacio total y libre:")
    subprocess.run(["df", "-h", "/runpod-volume"], check=False)
    print("\nPeso de las carpetas dentro de /runpod-volume/models:")
    subprocess.run(["du", "-sh", "/runpod-volume/models/SDXL"], check=False)
    subprocess.run(["du", "-sh", "/runpod-volume/models/Qwen-Image-Layered"], check=False)
    subprocess.run(["du", "-sh", "/runpod-volume/models/huggingface"], check=False)
    print("===========================================")

    print("Inicializando contenedor y cargando TUBERÍA DUAL en H100 (80GB VRAM)...")

    try:
        # 1. Cargar SDXL (El Artista) desde el directorio local para no descargarlo de nuevo
        print("Cargando SDXL Img2Img desde el Network Volume...")
        pipeline_sdxl = AutoPipelineForImage2Image.from_pretrained(
            SDXL_DIR,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            token=HF_TOKEN
        ).to("cuda")
        pipeline_sdxl.set_progress_bar_config(disable=True)
        
        # 2. Cargar Qwen-Image-Layered (El Cirujano - FRANKENSTEIN FP8)
        print("Cargando Qwen-Image-Layered (Transformer FP8 + Oficial)...")
        # Cargamos el archivo FP8 suelto solo como el motor del Transformer
        transformer_fp8 = QwenImageTransformer2DModel.from_single_file(
            QWEN_FILE,
            config="Qwen/Qwen-Image-Layered",
            subfolder="transformer",
            torch_dtype=torch.float8_e4m3fn,
            token=HF_TOKEN
        )
        
        # Juntamos el motor FP8 con las demás piezas (VAE, Tokenizer) del repo oficial ...
        pipeline_qwen = QwenImageLayeredPipeline.from_pretrained(
            "Qwen/Qwen-Image-Layered", 
            transformer=transformer_fp8,
            torch_dtype=torch.float16,
            token=HF_TOKEN
        ).to("cuda")
        pipeline_qwen.set_progress_bar_config(disable=True)
        
        print("¡Tubería Dual cargada exitosamente en VRAM!")
    except Exception as e:
        print(f"Advertencia Crítica: Fallo al cargar los modelos. Detalle: {e}")
        pipeline_sdxl = None
        pipeline_qwen = None

    # Cliente S3 para Cloudflare R2
    s3_client = boto3.client(
        service_name ="s3",
        endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id = R2_ACCESS_KEY,
        aws_secret_access_key = R2_SECRET_KEY,
        region_name="auto",
    )

    def process_and_upload_layer(layer_img: Image.Image, layer_index: int, job_id: str, width_cm: int, height_cm: int, dpi: int):
        # 1. Calcular tamaño en píxeles (Fórmula: Pixeles = (Centímetros / 2.54) * DPI)
        target_width_px = int((width_cm / 2.54) * dpi)
        target_height_px = int((height_cm / 2.54) * dpi)
        
        # 2. Redimensionar usando LANCZOS
        resized_img = layer_img.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)
        
        # 3. Guardar con metadata 300 DPI
        buffer = io.BytesIO()
        resized_img.save(buffer, format="PNG", dpi=(dpi, dpi))
        buffer.seek(0)
        
        # 4. Subir a R2
        file_key = f"jobs/{job_id}/layer_{layer_index}.png"
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
        
        if not pipeline_sdxl or not pipeline_qwen:
            return {"error": "Los modelos no están cargados correctamente. Revisa los logs de inicialización."}

        if not image_url:
            return {"error": "Se requiere una image_url base para la transformación Img2Img."}

        print(f"Job {job_id}: Procesando '{prompt}' a {print_width_cm}x{print_height_cm}cm ({print_dpi} DPI)")

        try:
            # Descargar imagen base
            response = requests.get(image_url)
            input_image = Image.open(io.BytesIO(response.content)).convert("RGB")
            
            # Inferencia Tubería Dual en la H100
            with torch.inference_mode():
                # PASO 1: Redibujar con SDXL (El Artista)
                sdxl_output = pipeline_sdxl(
                    prompt=prompt, 
                    image=input_image, 
                    strength=strength, 
                    guidance_scale=7.5,
                    num_inference_steps=40
                ).images[0]
                
                # PASO 2: Extraer capas con Qwen (El Cirujano)
                # Aseguramos que la salida de SDXL (RGB) se convierta a RGBA para Qwen
                qwen_inputs = {
                    "image": sdxl_output.convert("RGBA"),
                    "generator": torch.Generator(device='cuda').manual_seed(777),
                    "num_inference_steps": 30,
                    "layers": 4, 
                }
                qwen_output = pipeline_qwen(**qwen_inputs)
                output_image_layers = qwen_output.images[0]

            # PASO 3: Procesar, redimensionar, inyectar DPI y subir
            layer_urls = []
            for i, layer_img in enumerate(output_image_layers):
                url = process_and_upload_layer(layer_img, i, job_id, print_width_cm, print_height_cm, print_dpi)
                layer_urls.append({"name": f"layer_{i}.png", "url": url})
                
            return {
                "success": True,
                "message": "Transformación Img2Img + Extracción de Capas completada exitosamente.",
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

