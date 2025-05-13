import requests
import os
import glob

def download_reference_images(urls: list[str]):
    idx = 1

    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.abspath(os.path.join(base_dir, "../../ai/reference_samples/"))

    print("다운로드할 URL 리스트:", urls)
    print("저장 경로:", local_path)

    # 폴더 없으면 만들어주기
    os.makedirs(local_path, exist_ok=True)

    for url in urls:
        response = requests.get(url)
        image_name = f"reference{idx}.jpg"
        idx += 1

        with open(os.path.join(local_path, image_name), "wb") as f:
            f.write(response.content)

def download_test_image(url: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.abspath(os.path.join(base_dir, "../../ai/test_samples/"))

    print("다운로드할 URL:", url)
    print("저장 경로:", local_path)

    # 폴더 없으면 만들어주기
    os.makedirs(local_path, exist_ok=True)

    response = requests.get(url)
    image_name = "test.jpg"
    with open(os.path.join(local_path, image_name), "wb") as f:
        f.write(response.content)

def delete_reference_images():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.abspath(os.path.join(base_dir, "../../ai/reference_samples/"))
    image_paths = glob.glob(os.path.join(folder_path, "reference*.jpg"))
    
    for path in image_paths:
        try:
            os.remove(path)
            print(f"🗑️ 삭제 완료: {path}")
        except Exception as e:
            print(f"❌ 삭제 실패: {path} → {e}")

def delete_test_image():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.abspath(os.path.join(base_dir, "../../ai/test_samples/"))
    image_path = os.path.join(os.path.join(folder_path, "test.jpg"))
    
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"🗑️ 삭제 완료: {image_path}")
        except Exception as e:
            print(f"❌ 삭제 실패: {image_path} → {e}")
    else:
        print(f"⚠️ 파일 없음: {image_path}")
