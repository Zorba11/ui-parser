import modal
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
from typing import Optional
import traceback

# Create app and web app
app = modal.App("ui-coordinates-finder")
web_app = FastAPI()

# Add your model initialization to the app
@app.function(gpu="T4")
def init_models():
    from utils import get_yolo_model, get_caption_model_processor
    
    yolo_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
    caption_model_processor = get_caption_model_processor(
        model_name="florence2", 
        model_name_or_path="weights/icon_caption_florence"
    )
    return yolo_model, caption_model_processor

# Configure CORS
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.function(gpu="T4", timeout=300)
@web_app.post("/process")
async def process_image_endpoint(
    request: Request, 
    file: UploadFile = File(...),
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    screen_width: int = 1920,
    screen_height: int = 1080
):
    try:
        # Add logging for debugging
        print(f"Processing file: {file.filename}")
        
        # Read and process the image
        contents = await file.read()
        print("File read successfully")
        
        # Save image temporarily
        image_save_path = '/tmp/saved_image_demo.png'
        image = Image.open(io.BytesIO(contents))
        image.save(image_save_path)
        
        # Initialize models
        yolo_model, caption_model_processor = init_models()
        
        # Process with OCR and detection
        from utils import check_ocr_box, get_som_labeled_img
        
        draw_bbox_config = {
            'text_scale': 0.8,
            'text_thickness': 2,
            'text_padding': 2,
            'thickness': 2,
        }
        
        ocr_bbox_rslt, _ = check_ocr_box(
            image_save_path, 
            display_img=False, 
            output_bb_format='xyxy',
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_save_path,
            yolo_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            iou_threshold=iou_threshold
        )
        
        # Format the output similar to Gradio demo
        output_text = []
        for i, (element_id, coords) in enumerate(label_coordinates.items()):
            x, y, w, h = coords
            
            # Calculate center points (normalized)
            center_x_norm = x + (w/2)
            center_y_norm = y + (h/2)
            
            # Calculate screen coordinates
            screen_x = int(center_x_norm * screen_width)
            screen_y = int(center_y_norm * screen_height)
            screen_w = int(w * screen_width)
            screen_h = int(h * screen_height)
            
            if i < len(parsed_content_list):
                element_desc = parsed_content_list[i]
                output_text.append({
                    "description": element_desc,
                    "normalized_coords": (center_x_norm, center_y_norm),
                    "screen_coords": (screen_x, screen_y),
                    "dimensions": (screen_w, screen_h)
                })
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Success",
                "filename": file.filename,
                "processed_image": dino_labled_img,  # Base64 encoded image
                "elements": output_text
            }
        )
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error processing request: {error_details}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "details": error_details
            }
        )

@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app

if __name__ == "__main__":
    app.serve()