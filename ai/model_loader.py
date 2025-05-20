from tensorflow.keras.models import load_model
import tensorflow as tf

# ========================= 커스텀 레이어 =========================
class L1DistanceLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L1DistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(L1DistanceLayer, self).get_config()
        return config

# ========================= Contrastive Loss =========================
def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, tf.float32)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def build_and_load_model(model_path):
    print(f"모델 로드 중: {model_path}")
    custom_objects = {'L1DistanceLayer': L1DistanceLayer, 'contrastive_loss': contrastive_loss}
    model = load_model(model_path, custom_objects=custom_objects)
    print("모델 로드 완료")
    return model
