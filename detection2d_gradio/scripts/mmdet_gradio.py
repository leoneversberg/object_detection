#!/usr/bin/env python3
import mmcv
import cv2
from mmdet.apis import init_detector, inference_detector
import gradio as gr
import numpy as np

def process(input_img):
    result = inference_detector(model, input_img)
    bboxes = result[0]
    bbox=bboxes[0] # select bbox with highest confidence
    x1, y1, x2, y2, conf = bbox
    color = (255, 0, 0)

    #start_point = (det2d.bbox.center.x - int(det2d.bbox.size_x/2), det2d.bbox.center.y-int(det2d.bbox.size_y/2))
    #end_point = (det2d.bbox.center.x + int(det2d.bbox.size_x/2), det2d.bbox.center.y+int(det2d.bbox.size_y/2))

    start_point = (int(x1), int(y1))
    end_point = (int(x2), int(y2))
    img = cv2.rectangle(input_img, start_point, end_point, color, 3)
    #img = cv2.putText(img, str(round(conf,2)), start_point, cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    return img

# init model
config_file = '/app/models/mmdetection_work_dir/faster_rcnn_config.py'
checkpoint_file = '/app/models/mmdetection_work_dir/latest.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# gradio
#examples = [["/app/examples/real1.jpg"], ["/app/examples/real2.jpg"], ["/app/examples/synthetic1.jpg"],["/app/examples/synthetic2.jpg"]]
demo = gr.Interface(process, inputs="image", outputs="image", title="Object Localization Model", description="This model was trained on 5000 synthetic jpg images with size 640 x 480 pixel", live=False)

if __name__ == "__main__":
    #print("visit http://localhost:7860 in your browser", flush=True)
    demo.launch(share=True)