import os
import shutil
import glob
import requests
import pytest

from app.api.s3_manager import (
    download_reference_images,
    download_test_image,
    delete_reference_images,
    delete_test_image,
)

TEST_DIR = "../../ai/test_samples/"

def test_download_test_image():
    image_url = "https://ggzz-img.s3.ap-northeast-2.amazonaws.com/verification/%E1%84%80%E1%85%A5%E1%86%B7%E1%84%8C%E1%85%B3%E1%86%BC%E1%84%86%E1%85%AE%E1%86%AF1.png"
    download_test_image(image_url)

    assert os.path.exists(os.path.join(TEST_DIR, "test.jpg"))

def test_delete_test_image():
    path = os.path.join(TEST_DIR, "test.jpg")
    with open(path, "wb") as f:
        f.write(b"test")

    delete_test_image()
    assert not os.path.exists(path)