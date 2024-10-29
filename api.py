from fastapi import FastAPI, File, UploadFile, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/process")
@limiter.limit("5/minute")
async def process_image(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Your processing logic here
        return JSONResponse(
            status_code=200,
            content={"message": "Success", "filename": file.filename}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        ) 