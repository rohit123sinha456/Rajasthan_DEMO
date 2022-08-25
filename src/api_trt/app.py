import os
import shutil
import logging
import time
from typing import Optional, List
import msgpack

from fastapi import FastAPI, File, Form, UploadFile, Header
from fastapi.encoders import jsonable_encoder
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse, RedirectResponse, PlainTextResponse
from fastapi.responses import UJSONResponse
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)

from modules.processing import Processing
from env_parser import EnvConfigs
from schemas import BodyDraw, BodyExtract

__version__ = "0.7.3.0"

dir_path = os.path.dirname(os.path.realpath(__file__))

# Read runtime settings from environment variables
configs = EnvConfigs()

logging.basicConfig(
    level=configs.log_level,
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)

processing = Processing(det_name=configs.models.det_name, rec_name=configs.models.rec_name,
                        ga_name=configs.models.ga_name,
                        mask_detector=configs.models.mask_detector,
                        device=configs.models.device,
                        max_size=configs.defaults.max_size,
                        max_rec_batch_size=configs.models.rec_batch_size,
                        max_det_batch_size=configs.models.det_batch_size,
                        backend_name=configs.models.backend_name,
                        force_fp16=configs.models.fp16,
                        triton_uri=configs.models.triton_uri,
                        root_dir='/models'
                        )

app = FastAPI(
    title="Rajasthan API",
    description="FastAPI wrapper for Rajasthan API.",
    version=__version__,
    docs_url=None,
    redoc_url=None
)

UPLOAD_FOLDER = "/app/uploads/"
PROCESSED_FOLDER = "/app/processed/"
file_processed_flag = False
FILENAME = ""


@app.post('/upload_video', tags=['Detection'])
def read_video(file: UploadFile = File(...)):
    global FILENAME
    FILENAME = file.filename
    if(os.path.exists(UPLOAD_FOLDER+file.filename)):
        return{"message":"File Exists"}
    with open(f'{UPLOAD_FOLDER+file.filename}', "wb") as buffer:
        shutil.copyfileobj(file.file,buffer)
    FILENAME = file.filename
    return {"message":"File Created"}

@app.post('/video_detections', tags=['Detection'])
async def draw_video():
    global file_processed_flag
    global FILENAME
    output = await processing.draw_faces_in_videos(os.path.join(UPLOAD_FOLDER,FILENAME),PROCESSED_FOLDER,FILENAME)
    file_processed_flag = output
    return {"message":"File Processing of "+FILENAME +" is successful"+str(output)}



@app.post('/image_recognition', tags=['Image recognition'])
async def image_recog(files: List[UploadFile]):
    global file_processed_flag
    if(len(files) != 2):
        return{"message":"Please input two files one of target and one of the source"}
    for file in files:
        if(os.path.exists(UPLOAD_FOLDER+file.filename)):
            return{"message":"File Exists"}
        with open(f'{UPLOAD_FOLDER+file.filename}', "wb") as buffer:
            shutil.copyfileobj(file.file,buffer)
    output = await processing.extract_from_image(UPLOAD_FOLDER,PROCESSED_FOLDER,files)
    file_processed_flag = output
    return {"message":"File Processing of "+FILENAME +" is successful"+str(output)}



@app.post('/video_recognition', tags=['Video recognition'])
async def video_recognition(files: List[UploadFile]):
    t0 = time.time()
    global file_processed_flag
    global FILENAME
    FILENAME = files[1].filename
    if(len(files) != 2):
        return{"message":"Please input two files one of target and one of the source"}
    for file in files:
        if(os.path.exists(UPLOAD_FOLDER+file.filename)):
            return{"message":"File Exists"}
        with open(f'{UPLOAD_FOLDER+file.filename}', "wb") as buffer:
            shutil.copyfileobj(file.file,buffer)
    took = time.time() - t0
    logging.info("Target and source successfully written to disk in "+str(took*1000)+" ms")
    output = await processing.extract_from_video(UPLOAD_FOLDER,PROCESSED_FOLDER,files)
    file_processed_flag = output
    return {"message":"File Processing of "+FILENAME +" is successful"+str(output)}



@app.get('/download_video', tags=['Download'])
def read_video():
    global file_processed_flag
    global FILENAME
    file_path = os.path.join(PROCESSED_FOLDER,FILENAME)
    print(file_path)
    if(file_processed_flag):
        return FileResponse(file_path,media_type="video/mp4")
    else:
        return {"message":"File Still Processing"}

@app.get('/download_image', tags=['Download'])
def read_video():
    global file_processed_flag
    file_path = os.path.join(PROCESSED_FOLDER,"output.jpg")
    print(file_path)
    if(file_processed_flag):
        return FileResponse(file_path,media_type="image/jpeg")
    else:
        return {"message":"File Still Processing"}





@app.get('/clear_storage')
def clear_storage():
    try:
        dir = '/app/uploads'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
        dir1 = '/app/processed'
        for f in os.listdir(dir1):
            os.remove(os.path.join(dir1, f))
    except:
        return{"message":"Failed to delete the contents "}
    return{"message":"Successfully deleted the files of "+dir+" and "+dir1}

@app.get('/info', tags=['Utility'])
def info():
    """
    Enslist container configuration.

    """

    about = dict(
        version=__version__,
        tensorrt_version=os.getenv('TRT_VERSION', os.getenv('TENSORRT_VERSION')),
        log_level=configs.log_level,
        models=vars(configs.models),
        defaults=vars(configs.defaults),
    )
    about['models'].pop('ga_ignore', None)
    about['models'].pop('rec_ignore', None)
    about['models'].pop('mask_ignore', None)
    about['models'].pop('device', None)
    return about


@app.get('/', include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url='/static/favicon.png'
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )

