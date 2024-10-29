# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
from PIL import Image
from utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img
)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        self.yolo_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
        self.caption_model_processor = get_caption_model_processor(
            model_name="florence2", 
            model_name_or_path="weights/icon_caption_florence"
        )
        self.draw_bbox_config = {
            'text_scale': 0.8,
            'text_thickness': 2,
            'text_padding': 2,
            'thickness': 2,
        }

    def predict(
        self,
        image: Path = Input(description="Screenshot of the screen"),
        screen_width: int = Input(
            description="Screen width in pixels",
            default=1920,
            ge=800,  # Setting minimum reasonable screen width
            le=7680,  # Supporting up to 8K displays
        ),
        screen_height: int = Input(
            description="Screen height in pixels",
            default=1080,
            ge=600,  # Setting minimum reasonable screen height
            le=4320,  # Supporting up to 8K displays
        ),
        box_threshold: float = Input(
            description="Confidence threshold for box detection",
            default=0.05,
            ge=0.01,
            le=1.0,
        ),
        iou_threshold: float = Input(
            description="IOU threshold for overlap detection",
            default=0.1,
            ge=0.01,
            le=1.0,
        ),
    ) -> dict:
        """Run object detection on a screenshot and return coordinates"""
        
        # Ensure the input image exists and is valid
        if not image.exists():
            raise ValueError("Input image file does not exist")
            
        # Open and validate the image
        try:
            input_image = Image.open(image)
            input_image.verify()  # Verify it's a valid image
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")

        # Save input image temporarily
        image_save_path = '/tmp/input_image.png'
        input_image = Image.open(image)
        input_image.save(image_save_path)

        # Get OCR results
        ocr_bbox_rslt, _ = check_ocr_box(
            image_save_path, 
            display_img=False, 
            output_bb_format='xyxy',
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}
        )
        text, ocr_bbox = ocr_bbox_rslt

        # Get labeled image and coordinates
        dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_save_path,
            self.yolo_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=self.draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=text,
            iou_threshold=iou_threshold
        )

        # Format output
        elements = []
        for i, (element_id, coords) in enumerate(label_coordinates.items()):
            x, y, w, h = coords
            
            # Calculate center points (normalized)
            center_x_norm = x + (w/2)
            center_y_norm = y + (h/2)
            
            # Calculate screen coordinates
            screen_x = int(center_x_norm * screen_width)
            screen_y = int(center_y_norm * screen_height)
            
            # Calculate element dimensions on screen
            screen_w = int(w * screen_width)
            screen_h = int(h * screen_height)

            element = {
                "description": parsed_content_list[i] if i < len(parsed_content_list) else f"Icon {i}",
                "normalized_coordinates": {
                    "x": center_x_norm,
                    "y": center_y_norm
                },
                "screen_coordinates": {
                    "x": screen_x,
                    "y": screen_y
                },
                "dimensions": {
                    "width": screen_w,
                    "height": screen_h
                }
            }
            elements.append(element)

        return {
            "image": dino_labeled_img,  # Base64 encoded image
            "elements": elements
        }
