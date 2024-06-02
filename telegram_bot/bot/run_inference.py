import os

import torch
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

def run_yolo(weights, source, file_id, img_size=640):
    model = torch.hub.load(os.getenv('YOLO_DIR'), 'custom', path=weights, source='local')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    results = model(source, size=img_size)

    save_dir = Path(f'inferences')
    results.print()
    results.save(save_dir=save_dir, exist_ok=True)

    return save_dir
