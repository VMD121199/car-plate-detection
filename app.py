import sys

sys.path.append("../")
from fastapi import FastAPI, Depends
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # add code
    return {"prediction": file.filename}
