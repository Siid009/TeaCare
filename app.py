
from flask import Flask, render_template, request, jsonify
import os
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

API_KEY = os.getenv("API_KEY")

app = Flask(__name__)

TF_AVAILABLE = False
model = None

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import *
    from tensorflow.keras import backend as K
    TF_AVAILABLE = True
except ModuleNotFoundError:
    tf = None
    image = None
    preprocess_input = None
    Model = None
    np = None
    print("TensorFlow is not available; prediction is disabled.")


# -------------------- CBAM BLOCK --------------------
def cbam_block(feature_map, ratio=8):
    channel = feature_map.shape[-1]

    # Channel Attention
    avg_pool = GlobalAveragePooling2D()(feature_map)
    avg_pool = Dense(channel // ratio, activation='relu')(avg_pool)
    avg_pool = Dense(channel)(avg_pool)

    max_pool = GlobalMaxPooling2D()(feature_map)
    max_pool = Dense(channel // ratio, activation='relu')(max_pool)
    max_pool = Dense(channel)(max_pool)

    channel_attention = Activation('sigmoid')(Add()([avg_pool, max_pool]))
    channel_attention = Reshape((1,1,channel))(channel_attention)
    feature_map = Multiply()([feature_map, channel_attention])

    # Spatial Attention
    avg_spatial = Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(feature_map)
    max_spatial = Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(feature_map)
    concat = Concatenate(axis=3)([avg_spatial, max_spatial])
    spatial_attention = Conv2D(filters=1, kernel_size=7, padding='same',
                               activation='sigmoid')(concat)
    refined_feature = Multiply()([feature_map, spatial_attention])

    return refined_feature


# -------------------- INCEPTION MODULE --------------------
def inception_module(x, filters):
    f1, f3_in, f3_out, f5_in, f5_out, pool_proj = filters

    conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(x)

    conv3 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(x)
    conv3 = Conv2D(f3_out, (3,3), padding='same', activation='relu')(conv3)

    conv5 = Conv2D(f5_in, (1,1), padding='same', activation='relu')(x)
    conv5 = Conv2D(f5_out, (5,5), padding='same', activation='relu')(conv5)

    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    pool = Conv2D(pool_proj, (1,1), padding='same', activation='relu')(pool)

    output = concatenate([conv1, conv3, conv5, pool], axis=3)
    return output


# -------------------- GOOGLENET + CBAM --------------------
def GoogLeNet_CBAM(input_shape=(224,224,3), num_classes=6):
    inp = Input(shape=input_shape)

    x = Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu')(inp)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    x = Conv2D(64, (1,1), padding='same', activation='relu')(x)
    x = Conv2D(192, (3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    x = inception_module(x, (64, 96, 128, 16, 32, 32))
    x = inception_module(x, (128, 128, 192, 32, 96, 64))
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    x = cbam_block(x)

    x = inception_module(x, (192, 96, 208, 16, 48, 64))
    x = inception_module(x, (160, 112, 224, 24, 64, 64))
    x = inception_module(x, (128, 128, 256, 24, 64, 64))
    x = inception_module(x, (112, 144, 288, 32, 64, 64))
    x = inception_module(x, (256, 160, 320, 32, 128, 128))
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    x = cbam_block(x)

    x = inception_module(x, (256, 160, 320, 32, 128, 128))
    x = inception_module(x, (384, 192, 384, 48, 128, 128))

    x = cbam_block(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    return model



# Load model when TensorFlow is available
if TF_AVAILABLE:
    try:
        model = GoogLeNet_CBAM()
        model.load_weights("best_attention_model.h5")
    except Exception as e:
        print("TensorFlow model failed to load:", e)
        model = None

# Example class labels (change according to your dataset)
idx_to_class = {
    0: "algal_spot",
    1: "brown_blight",
    2: "gray_blight",
    3: "healthy",
    4: "helopeltis",
    5: "red_spot"
}

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



def get_smart_remedy(label, confidence):

    # Severity logic
    if confidence < 70:
        severity = "Mild"
    elif confidence < 90:
        severity = "Moderate"
    else:
        severity = "Severe"

    remedies = {
        "brown_blight": {
            "treatment": "Apply copper fungicide regularly",
            "organic": "Neem oil spray weekly",
            "chemical": "Copper oxychloride",
            "recovery": "2-3 weeks"
        },
        "red_spot": {
            "treatment": "Remove infected leaves",
            "organic": "Baking soda spray",
            "chemical": "Chlorothalonil",
            "recovery": "1-2 weeks"
        },
        "gray_blight": {
            "treatment": "Improve drainage",
            "organic": "Compost tea spray",
            "chemical": "Mancozeb",
            "recovery": "2 weeks"
        },
        "healthy": {
            "treatment": "No disease detected",
            "organic": "Maintain soil health",
            "chemical": "None",
            "recovery": "Healthy"
        }
    }

    info = remedies.get(label, {
        "treatment": "Consult expert",
        "organic": "Neem solution",
        "chemical": "General fungicide",
        "recovery": "Varies"
    })

    return severity, info


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    label = idx_to_class[pred_idx]
    confidence = float(preds[0][pred_idx] * 100)

    return label, confidence


def generate_leaf_visuals(img_path):
    """
    Build additional visual outputs for UI:
    - masked image: leaf foreground isolated
    - highlighted image: likely diseased zones emphasized
    """
    base = os.path.splitext(os.path.basename(img_path))[0]
    masked_name = f"{base}_masked.png"
    highlighted_name = f"{base}_highlighted.png"
    masked_path = os.path.join(UPLOAD_FOLDER, masked_name)
    highlighted_path = os.path.join(UPLOAD_FOLDER, highlighted_name)

    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)

    # Basic leaf segmentation: green-dominant pixels.
    leaf_mask = ((g > r * 0.9) & (g > b * 0.9) & (g > 40)).astype(np.uint8)
    if leaf_mask.mean() < 0.04:
        # Fallback for difficult lighting/backgrounds.
        gray = (0.299 * r + 0.587 * g + 0.114 * b)
        threshold = np.percentile(gray, 45)
        leaf_mask = (gray > threshold).astype(np.uint8)

    masked = arr.copy()
    masked[leaf_mask == 0] = np.array([247, 240, 240], dtype=np.uint8)
    Image.fromarray(masked).save(masked_path)

    # Likely disease region: brown/red/dark irregular areas within leaf.
    disease_mask = (
        ((r > g * 1.12) & (r > b * 1.05) & (r > 65)) |
        ((r < 120) & (g < 120) & (b < 120))
    ) & (leaf_mask == 1)

    highlighted = arr.copy()
    highlight_color = np.array([242, 181, 11], dtype=np.uint8)
    alpha = 0.55
    highlighted[disease_mask] = (
        (1 - alpha) * highlighted[disease_mask] + alpha * highlight_color
    ).astype(np.uint8)
    Image.fromarray(highlighted).save(highlighted_path)

    return masked_path, highlighted_path


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if not TF_AVAILABLE or model is None:
            error = (
                "Prediction is currently unavailable because TensorFlow is not installed "
                "or the model failed to load. Install TensorFlow and restart the app."
            )
            return render_template("predict.html", error=error)

        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            label, confidence = predict_image(filepath)
            severity, remedy = get_smart_remedy(label, confidence)
            masked_path, highlighted_path = generate_leaf_visuals(filepath)

            return render_template("predict.html",
                                   prediction=label,
                                   confidence=round(confidence, 2),
                                   severity=severity,
                                   remedy=remedy,
                                   image_path=filepath,
                                   masked_path=masked_path,
                                   highlighted_path=highlighted_path)

    return render_template("predict.html")




@app.route("/weather-page")
def weather_page():
    return render_template("weather.html")



@app.route("/weather", methods=["POST"])
def weather():
    try:
        data = request.json
        lat = data.get("lat")
        lon = data.get("lon")
        if lat is None or lon is None:
            return jsonify({"error": "Missing coordinates"}), 400

        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
            "precipitation,weather_code,wind_speed_10m"
            "&daily=weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max,uv_index_max,wind_speed_10m_max,"
            "relative_humidity_2m_mean"
            "&forecast_days=10&timezone=auto"
        )

        response = requests.get(url, timeout=15)
        res = response.json()
        if response.status_code != 200 or "current" not in res or "daily" not in res:
            return jsonify({
                "temp": "N/A",
                "humidity": "N/A",
                "condition": "API Error",
                "risk": "⚠️ Unable to detect risk",
                "advice": "Weather data unavailable",
                "forecast": []
            })

        current = res["current"]
        daily = res["daily"]

        temp = float(current.get("temperature_2m", 0))
        humidity = float(current.get("relative_humidity_2m", 0))
        wind = float(current.get("wind_speed_10m", 0))
        weather_code = int(current.get("weather_code", 0))
        uv_today = float(daily.get("uv_index_max", [0])[0] or 0)
        rain_prob_today = int(daily.get("precipitation_probability_max", [0])[0] or 0)

        code_map = {
            0: "Clear",
            1: "Mainly Clear",
            2: "Partly Cloudy",
            3: "Cloudy",
            45: "Fog",
            48: "Fog",
            51: "Drizzle",
            53: "Drizzle",
            55: "Drizzle",
            56: "Freezing Drizzle",
            57: "Freezing Drizzle",
            61: "Rain",
            63: "Rain",
            65: "Heavy Rain",
            66: "Freezing Rain",
            67: "Freezing Rain",
            71: "Snow",
            73: "Snow",
            75: "Snow",
            80: "Rain Showers",
            81: "Rain Showers",
            82: "Heavy Showers",
            95: "Thunderstorm",
            96: "Thunderstorm",
            99: "Thunderstorm",
        }
        condition = code_map.get(weather_code, "Moderate")

        # Disease risk
        if humidity >= 80 or rain_prob_today >= 70:
            disease_risk = "High"
            risk = "⚠️ High fungal disease risk"
        elif humidity >= 65 or rain_prob_today >= 45:
            disease_risk = "Moderate"
            risk = "⚠️ Moderate disease risk"
        else:
            disease_risk = "Low"
            risk = "✅ Low disease risk"

        # Crop impact and irrigation
        if temp > 32:
            crop_impact = "Heat stress likely. Tender leaves may scorch during peak sun."
            irrigation = "Irrigate early morning and add temporary shade near young plants."
        elif temp < 14:
            crop_impact = "Cool stress may reduce growth speed and flushing."
            irrigation = "Keep soil lightly moist; avoid overwatering in cold periods."
        else:
            crop_impact = "Temperature range is acceptable for tea growth."
            irrigation = "Maintain moderate moisture and avoid waterlogging."

        if rain_prob_today >= 60:
            rain_alert = "Rain alert active: postpone foliar spray and improve drainage."
        else:
            rain_alert = "No strong rain alert: spray windows are relatively safer."

        if wind > 20:
            spray_safety = "Caution: wind is high for spraying."
        else:
            spray_safety = "Spray-safe wind conditions."

        if uv_today >= 8:
            uv_risk = "High UV: monitor leaf burn risk."
        elif uv_today >= 5:
            uv_risk = "Moderate UV: midday stress possible."
        else:
            uv_risk = "Low UV stress today."

        advice = (
            "Use early-morning monitoring and target treatment blocks with highest humidity first."
        )

        forecast = []
        dates = daily.get("time", [])
        d_codes = daily.get("weather_code", [])
        d_tmax = daily.get("temperature_2m_max", [])
        d_tmin = daily.get("temperature_2m_min", [])
        d_rain = daily.get("precipitation_probability_max", [])
        d_uv = daily.get("uv_index_max", [])
        d_wind = daily.get("wind_speed_10m_max", [])
        d_hum = daily.get("relative_humidity_2m_mean", [])

        for i in range(min(10, len(dates))):
            code_i = int(d_codes[i]) if i < len(d_codes) else 0
            forecast.append({
                "date": dates[i],
                "condition": code_map.get(code_i, "Moderate"),
                "weather_code": code_i,
                "temp_max": round(float(d_tmax[i]), 1) if i < len(d_tmax) else None,
                "temp_min": round(float(d_tmin[i]), 1) if i < len(d_tmin) else None,
                "rain_probability": int(d_rain[i]) if i < len(d_rain) and d_rain[i] is not None else 0,
                "uv": round(float(d_uv[i]), 1) if i < len(d_uv) and d_uv[i] is not None else 0,
                "wind": round(float(d_wind[i]), 1) if i < len(d_wind) and d_wind[i] is not None else 0,
                "humidity": int(d_hum[i]) if i < len(d_hum) and d_hum[i] is not None else 0
            })

        return jsonify({
            "temp": round(temp, 1),
            "humidity": int(humidity),
            "wind": round(wind, 1),
            "uv": round(uv_today, 1),
            "condition": condition,
            "risk": risk,
            "disease_risk": disease_risk,
            "advice": advice,
            "crop_impact": crop_impact,
            "irrigation": irrigation,
            "rain_alert": rain_alert,
            "spray_safety": spray_safety,
            "uv_risk": uv_risk,
            "forecast": forecast
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({
            "temp": "Error",
            "humidity": "Error",
            "condition": "Server error",
            "risk": "⚠️ Try again",
            "advice": "⚠️ No advice available",
            "forecast": []
        })

if __name__ == "__main__":
    app.run(debug=True)
