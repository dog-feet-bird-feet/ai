import requests
import os
import glob

from image_type import Type

def download_reference_images(urls: list[str]):

    idx = 1
    for url in urls:
        response = requests.get(url)
        local_path = "/ai/refence_samples/"
        image_name = "refenrence" + idx + ".jpg"
        idx+=1

        with open(local_path + image_name, "wb") as f:
            f.write(response.content)

def download_test_image(url: str):
    response = requests.get(url)
    local_path = "/ai/test_samples/"
    image_name = "test" + ".jpg"

    with open(local_path + image_name, "wb") as f:
        f.write(response.content)

def delete_reference_images():
    folder_path = "/ai/refence_samples/"
    image_paths = glob.glob(os.path.join(folder_path, "refenrence*.jpg"))
    
    for path in image_paths:
        try:
            os.remove(path)
            print(f"🗑️ 삭제 완료: {path}")
        except Exception as e:
            print(f"❌ 삭제 실패: {path} → {e}")

def delete_test_image():
    folder_path = "/ai/test_samples/"
    image_path = os.path.join(folder_path, "test.jpg")
    
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"🗑️ 삭제 완료: {image_path}")
        except Exception as e:
            print(f"❌ 삭제 실패: {image_path} → {e}")
    else:
        print(f"⚠️ 파일 없음: {image_path}")
