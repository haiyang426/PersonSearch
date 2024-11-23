from PIL import Image, ImageDraw

def draw_rectangle(image_path, bbox, save_path):
    # 加载图像
    x, y, w, h = bbox
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # 画矩形框，xywh 转换为矩形的两个角的坐标
    rectangle = [x, y, x + w, y + h]
    draw.rectangle(rectangle, outline="red", width=2)
    
    # 保存图像
    img.save(save_path)

# 示例用法
draw_rectangle("/data/zhy/person_search_with_mmdetection/data/CUHK-SYSU/Image/SSM/s13246.jpg", [596,  62, 124, 381], "vis.jpg")
