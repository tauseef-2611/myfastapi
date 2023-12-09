from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from pathlib import Path
from model import monaimodel

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Global variable to store the uploaded file content
global_file_content = None

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    global global_file_content
    global_file_content = await file.read()
    print("file read")
    # Save the file with a specific filename, such as "example.obj"
    #filename = global_file_content.filename
    with open("my.nii.gz", "wb") as f:
         f.write(global_file_content)
    print("file saved")
    monaimodel.predict("my.nii.gz")
    print("Obj saved")
    return {"filename": "good"}

@app.post("/test/")
async def test_api():
    return {"test": "good"}

@app.get("/get_obj_fil")
async def get_uploaded_file():
    global global_file_content

    if global_file_content is None:
        raise HTTPException(status_code=404, detail="No file uploaded")

    try:
        return StreamingResponse(io.BytesIO(global_file_content), media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename=uploaded_file.obj"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/get_obj_file")
def get_obj_file():
    file_path=Path("obj_pred.obj")
    #file_path = Path("E:/KMIT/3D VR Radiology/myobj.obj")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, media_type="application/octet-stream", filename="example.obj")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
