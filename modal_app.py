import modal
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Create app
app = modal.App("ui-coordinates-finder")

# Create FastAPI app
web_app = FastAPI()

# Configure CORS
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@web_app.post("/process")
async def process_image_endpoint(request: Request, file: UploadFile = File(...)):
    try:
        print("Received request")
        contents = await file.read()
        print(f"Processing file: {file.filename}")
        # Your processing logic here
        return JSONResponse(
            status_code=200,
            content={"message": "Success", "filename": file.filename}
        )
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Mount the FastAPI app
@modal.asgi_app()
def fastapi_app():
    return web_app

if __name__ == "__main__":
    modal.serve(fastapi_app)