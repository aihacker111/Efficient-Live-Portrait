import os
import requests
from tqdm import tqdm
from pathlib import Path


def downloading(url, outf):
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(outf, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print(f"Downloaded successfully to {outf}")
    else:
        print(f"{outf} already exists.")
        return outf


MODEL_URLS = {
    'yolox_l_pose': 'https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/yolox_l.onnx?download=true',
    'dw_pose': 'https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/dw-ll_ucoco_384.onnx?download=true'
}


def get_open_pose_model():
    # Download the models and save them in the current working directory
    current_dir = os.getenv('DW_POSE', default=str(Path.home()))
    model_dir = os.path.join(current_dir, 'dw_pose_weights')
    os.makedirs(model_dir, exist_ok=True)

    model_paths = {}
    for model_name, url in MODEL_URLS.items():
        filename = url.split('/')[-1].split('?')[0]
        save_path = os.path.join(model_dir, filename)
        downloading(url, save_path)
        model_paths[model_name] = save_path

    print(f'Downloaded successfully and already saved: {model_dir}')
    return model_paths, model_dir
