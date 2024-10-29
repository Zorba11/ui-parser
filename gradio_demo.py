from typing import Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image
import io


import base64, os
from utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
import torch
from PIL import Image

yolo_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")
platform = 'pc'
if platform == 'pc':
    draw_bbox_config = {
        'text_scale': 0.8,
        'text_thickness': 2,
        'text_padding': 2,
        'thickness': 2,
    }
elif platform == 'web':
    draw_bbox_config = {
        'text_scale': 0.8,
        'text_thickness': 2,
        'text_padding': 3,
        'thickness': 3,
    }
elif platform == 'mobile':
    draw_bbox_config = {
        'text_scale': 0.8,
        'text_thickness': 2,
        'text_padding': 3,
        'thickness': 3,
    }



MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""

DEVICE = torch.device('cuda')

# @spaces.GPU
# @torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process(
    image_input,
    box_threshold,
    iou_threshold,
    screen_width,
    screen_height
) -> Optional[Image.Image]:
    """
    Process the image and return both normalized and screen coordinates
    
    Args:
        image_input: Input image
        box_threshold: Confidence threshold for box detection
        iou_threshold: IOU threshold for overlap detection
        screen_width: Actual screen width in pixels
        screen_height: Actual screen height in pixels
    """
    image_save_path = 'imgs/saved_image_demo.png'
    image_input.save(image_save_path)

    # Get image dimensions
    image_width = image_input.width
    image_height = image_input.height

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_save_path, display_img=False, output_bb_format='xyxy', 
                                                   goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9})
    text, ocr_bbox = ocr_bbox_rslt
    
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_save_path, yolo_model, BOX_TRESHOLD=box_threshold, 
        output_coord_in_ratio=True, ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config, 
        caption_model_processor=caption_model_processor, 
        ocr_text=text, iou_threshold=iou_threshold
    )
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    
    # Format the output to include both normalized and screen coordinates
    output_text = []
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
        
        if i < len(parsed_content_list):
            # For text elements
            element_desc = parsed_content_list[i]
            output_text.append(
                f"{element_desc}\n"
                f"  Normalized coordinates: ({center_x_norm:.3f}, {center_y_norm:.3f})\n"
                f"  Screen coordinates: ({screen_x}, {screen_y})\n"
                f"  Dimensions: {screen_w}x{screen_h} pixels"
            )
        else:
            # For icon elements without text
            output_text.append(
                f"Icon {i}\n"
                f"  Normalized coordinates: ({center_x_norm:.3f}, {center_y_norm:.3f})\n"
                f"  Screen coordinates: ({screen_x}, {screen_y})\n"
                f"  Dimensions: {screen_w}x{screen_h} pixels"
            )
    
    parsed_content = '\n\n'.join(output_text)
    return image, parsed_content



with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image_input_component = gr.Image(
                type='pil', label='Upload image')
            
            with gr.Row():
                # Screen dimension inputs
                screen_width_component = gr.Number(
                    label='Screen Width (pixels)', 
                    value=1920,  # Default value
                    precision=0
                )
                screen_height_component = gr.Number(
                    label='Screen Height (pixels)', 
                    value=1080,  # Default value
                    precision=0
                )
            
            # Threshold sliders
            box_threshold_component = gr.Slider(
                label='Box Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.05)
            iou_threshold_component = gr.Slider(
                label='IOU Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.1)
            
            submit_button_component = gr.Button(
                value='Submit', variant='primary')
                
        with gr.Column():
            image_output_component = gr.Image(type='pil', label='Image Output')
            text_output_component = gr.Textbox(
                label='Parsed screen elements', 
                placeholder='Text Output',
                lines=10  # Increased to show more content
            )

    submit_button_component.click(
        fn=process,
        inputs=[
            image_input_component,
            box_threshold_component,
            iou_threshold_component,
            screen_width_component,
            screen_height_component
        ],
        outputs=[image_output_component, text_output_component]
    )

# demo.launch(debug=False, show_error=True, share=True)
demo.launch(share=True, server_port=7861, server_name='0.0.0.0')
