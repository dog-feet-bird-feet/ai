import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import glob
from app.models.analyzeResponse import AnalyzeResponse

# ========================= 수동 특징 추출 =========================
def extract_handcrafted_features(gray_img, binary_img=None):
    features = []
    HANDCRAFTED_FEATURES_DIM = 12

    if binary_img is None:
        _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    pixel_density = np.sum(binary_img > 0) / binary_img.size
    features.append(pixel_density)

    contours, _ = cv2.findContours(binary_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles = []
    for contour in contours:
        if len(contour) > 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                angle = ellipse[2]
                if angle > 90:
                    angle -= 180
                angles.append(angle)
            except:
                pass

    mean_angle = np.mean(angles) if angles else 0
    std_angle = np.std(angles) if angles else 0
    features.append(mean_angle / 90)
    features.append(std_angle / 45)

    heights, widths, areas, aspect_ratios = [], [], [], []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 20:
            heights.append(h)
            widths.append(w)
            areas.append(area)
            aspect_ratios.append(w / h if h > 0 else 0)

    if heights and widths:
        features.extend([
            np.mean(heights) / 100,
            np.std(heights) / 50,
            np.mean(widths) / 100,
            np.std(widths) / 50,
            np.mean(areas) / 1000,
            np.mean(aspect_ratios),
            np.std(aspect_ratios)
        ])
    else:
        features.extend([0] * 7)

    features = features[:HANDCRAFTED_FEATURES_DIM]
    features.extend([0] * (HANDCRAFTED_FEATURES_DIM - len(features)))
    return np.array(features, dtype=np.float32)

# ========================= 이미지 전처리 =========================
def preprocess_image(image_path, target_height=64, target_width=512):
    try:
        print(f"✅ image_path = {image_path}")
        print(f"✅ os.path.isfile(image_path) = {os.path.isfile(image_path)}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        original_gray = gray.copy()

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

        h, w = binary.shape
        aspect = w / h
        if aspect >= target_width / target_height:
            new_width = target_width
            new_height = max(int(target_width / aspect), target_height // 2)
        else:
            new_height = target_height
            new_width = max(int(target_height * aspect), target_width // 2)

        resized = cv2.resize(binary, (new_width, new_height))
        canvas = np.zeros((target_height, target_width), dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        normalized = canvas.astype(np.float32) / 255.0
        expanded = np.expand_dims(normalized, axis=-1)
        if expanded.shape != (target_height, target_width, 1):
            expanded = np.reshape(expanded, (target_height, target_width, 1))

        handcrafted_features = extract_handcrafted_features(original_gray, binary)
        return expanded, handcrafted_features

    except Exception as e:
        print(f"이미지 전처리 오류 ({image_path}): {e}")
        return None, None

# ========================= 유사도 계산 =========================
def get_similarity(model, image1_path, image2_path):
    img1_result = preprocess_image(image1_path)
    img2_result = preprocess_image(image2_path)

    if img1_result[0] is None or img2_result[0] is None:
        return None, None, None

    img1, hand1 = img1_result
    img2, hand2 = img2_result

    img1_batch = np.expand_dims(img1, axis=0)
    hand1_batch = np.expand_dims(hand1, axis=0)
    img2_batch = np.expand_dims(img2, axis=0)
    hand2_batch = np.expand_dims(hand2, axis=0)

    similarity = model.predict([img1_batch, hand1_batch, img2_batch, hand2_batch])[0][0]

    pressure = (hand1[0] + hand2[0]) / 2
    slant = (hand1[1] + hand2[1]) / 2

    return similarity, pressure, slant

# ========================= 결과 생성 =========================
def create_result(results, avg_score):
    if not results:
        print("❌ 비교할 결과 없음")
        return None

    best_result = results[0]
    avg_pressure = best_result.get('pressure', 0.0)
    avg_slant = best_result.get('slant', 0.0)

    print("\n" + "=" * 50)
    print("📝 최종 결과 요약")
    print(f"📌 평균 유사도: {avg_score*100:.4f}%")
    print(f"📌 평균 필압: {avg_pressure:.4f}")
    print(f"📌 평균 기울기: {avg_slant:.4f}")
    print("=" * 50)

    return AnalyzeResponse(
        float(avg_score),
        float(avg_pressure),
        float(avg_slant),
        ""
    )

def find_first_image(folder_path):
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            return os.path.join(folder_path, fname)
    return None

def analyze(model):
    reference_folder = 'ai/reference_samples'
    test_image_path = find_first_image("ai/test_samples")

    similarity_scores = []

    for filename in os.listdir(reference_folder):
        ref_path = os.path.join(reference_folder, filename)
        if os.path.isfile(ref_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"✅ 유효한 이미지 파일: {filename}")
            similarity, pressure, slant = get_similarity(model, ref_path, test_image_path)
            if similarity is not None:
                similarity_scores.append({
                    'reference': filename,
                    'similarity': similarity,
                    'pressure': pressure,
                    'slant': slant
                })
        else:
            print(f"⚠️ 무시된 파일: {filename}")

    similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)

    if similarity_scores:
        avg_score = np.mean([item['similarity'] for item in similarity_scores])
        print("\n" + "#" * 50)
        print(f"🔍 전체 평균 유사도: {avg_score:.4f}")
        print(f"✔️ 비교한 이미지 수: {len(similarity_scores)}")
        print("#" * 50)

        threshold = 0.5
        print("#" * 50)
        if avg_score >= threshold:
            print(f"✅ 판별 결과: 같은 사람입니다 (유사도 ≥ {threshold})")
        else:
            print(f"❌ 판별 결과: 다른 사람입니다 (유사도 < {threshold})")
        print("#" * 50)

        summary = create_result(similarity_scores, avg_score)
        return summary
    else:
        print("❌ 유사도 계산에 실패했습니다.")
        return None



