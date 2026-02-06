# src/data/converters/image_converter.py
import numpy as np
from PIL import Image
import taichi as ti

class ImageConverter:
    @staticmethod
    def image_to_colorfield(image_path, target_shape):
        """
        convert image to colorField, strictly follow the rule of "top full side, center on the other side":
        - if target aspect ratio > image aspect ratio:
        - if target aspect ratio < image aspect ratio:
        """
        with Image.open(image_path) as img:
            img_rgb = img.convert('RGB')
            orig_w, orig_h = img_rgb.size
            target_w, target_h = target_shape

            if target_w * orig_h > orig_w * target_h:
                new_h = target_h
                new_w = round(orig_w * (target_h / orig_h))
                x_offset = (target_w - new_w) // 2
                y_offset = 0
            else:
                new_w = target_w
                new_h = round(orig_h * (target_w / orig_w))
                x_offset = 0
                y_offset = (target_h - new_h) // 2

            resized_img = img_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)

            canvas = Image.new('RGB', (target_w, target_h), (0, 0, 0))
            canvas.paste(resized_img, (x_offset, y_offset))

            img_array = np.array(canvas, dtype=np.float32) / 255.0
            img_array = img_array[::-1, :, :]
            img_array = np.transpose(img_array, (1, 0, 2))
            color_field = ti.Vector.field(3, float, shape=target_shape)
            color_field.from_numpy(img_array)

            return color_field
    
    @staticmethod
    def colorfield_to_image(color_field, output_path):
        img_array = color_field.to_numpy()
        img_array = np.transpose(img_array, (1, 0, 2))
        img_array = img_array[::-1, :, :]
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(output_path)