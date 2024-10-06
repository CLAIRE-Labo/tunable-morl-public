import numpy as np
from PIL import Image, ImageDraw, ImageFont


def add_text(video: np.ndarray, text: str, color: str) -> np.ndarray:
    if color == 'white':
        color = (255, 255, 255)
    elif color == 'black':
        color = (0, 0, 0)
    elif color == 'red':
        color = (255, 0, 0)
    elif color == 'green':
        color = (0, 255, 0)
    elif color == 'blue':
        color = (0, 0, 255)
    else:
        raise ValueError('Invalid color')

    font = ImageFont.load_default()

    for t in range(video.shape[0]):
        frame = Image.fromarray(video[t].astype('uint8'), 'RGB')
        draw = ImageDraw.Draw(frame)
        x = 10
        y = frame.height - 30
        draw.text((x, y), text, font=font, fill=color)

        video[t] = np.array(frame)

    return video
