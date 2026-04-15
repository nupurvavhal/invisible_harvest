from datetime import datetime, timezone
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import secrets
import hashlib
import binascii
from pymongo import DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Setup templates and static files
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def load_env_file(env_path: Path):
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(BASE_DIR / ".env")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "fruit_analyzer")
SESSION_SECRET = os.getenv("SESSION_SECRET", "change-this-secret")

mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[MONGODB_DB]
users_collection = db["users"]
predictions_collection = db["predictions"]

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

# Load trained model
MODEL = tf.keras.models.load_model(str(BASE_DIR / "model" / "fixedfruit_model_v2.keras"))
CLASS_NAMES = [
    "Apple_Fresh",
    "Apple_Rotten",
    "Banana_Fresh",
    "Banana_Rotten",
    "Orange_Fresh",
    "Orange_Rotten",
    "Strawberry_Fresh",
    "Strawberry_Rotten",
]


def hash_password(password: str, salt: bytes | None = None) -> str:
    if salt is None:
        salt = os.urandom(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return f"{binascii.hexlify(salt).decode()}${binascii.hexlify(hashed).decode()}"


def verify_password(password: str, stored_password: str) -> bool:
    try:
        salt_hex, hash_hex = stored_password.split("$", 1)
        salt = binascii.unhexlify(salt_hex)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
        return binascii.hexlify(digest).decode() == hash_hex
    except (ValueError, binascii.Error):
        return False


def current_user(request: Request):
    return request.session.get("user")


def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return tf.keras.applications.efficientnet.preprocess_input(img_array)


@app.on_event("startup")
async def setup_database():
    try:
        users_collection.create_index("email", unique=True)
        predictions_collection.create_index([("created_at", DESCENDING)])

        admin_email = os.getenv("ADMIN_EMAIL", "admin@fruitanalyzer.local").strip().lower()
        admin_password = os.getenv("ADMIN_PASSWORD", "admin123")

        if not users_collection.find_one({"email": admin_email}):
            users_collection.insert_one(
                {
                    "name": "Administrator",
                    "email": admin_email,
                    "password": hash_password(admin_password),
                    "role": "admin",
                    "created_at": datetime.now(timezone.utc),
                }
            )
    except PyMongoError as error:
        print(f"MongoDB setup warning: {error}")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if user.get("role") == "admin":
        return RedirectResponse("/admin", status_code=303)
    return templates.TemplateResponse("index.html", {"request": request, "user_name": user.get("name", "User")})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = current_user(request)
    if user:
        if user.get("role") == "admin":
            return RedirectResponse("/admin", status_code=303)
        return RedirectResponse("/", status_code=303)

    registered = request.query_params.get("registered") == "1"
    return templates.TemplateResponse("login.html", {"request": request, "registered": registered, "error": None})


@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, email: str = Form(...), password: str = Form(...), role: str = Form(...)):
    normalized_email = email.strip().lower()
    selected_role = "admin" if role == "admin" else "user"

    try:
        user = users_collection.find_one({"email": normalized_email})
    except PyMongoError:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "registered": False, "error": "Database connection issue. Try again."},
            status_code=500,
        )

    if not user or not verify_password(password, user.get("password", "")):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "registered": False, "error": "Invalid email or password."},
            status_code=401,
        )

    if user.get("role") != selected_role:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "registered": False, "error": "Role does not match this account."},
            status_code=403,
        )

    request.session["user"] = {
        "name": user.get("name", "User"),
        "email": user.get("email", ""),
        "role": user.get("role", "user"),
    }

    if request.session["user"]["role"] == "admin":
        return RedirectResponse("/admin", status_code=303)
    return RedirectResponse("/", status_code=303)


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    if current_user(request):
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse("register.html", {"request": request, "error": None})


@app.post("/register", response_class=HTMLResponse)
async def register(request: Request, name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    normalized_name = name.strip()
    normalized_email = email.strip().lower()

    if len(password) < 6:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Password must be at least 6 characters."},
            status_code=400,
        )

    try:
        users_collection.insert_one(
            {
                "name": normalized_name,
                "email": normalized_email,
                "password": hash_password(password),
                "role": "user",
                "created_at": datetime.now(timezone.utc),
            }
        )
    except DuplicateKeyError:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Email already exists."},
            status_code=409,
        )
    except PyMongoError:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Database error. Please try again."},
            status_code=500,
        )

    return RedirectResponse("/login?registered=1", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    user = current_user(request)
    if not user:
        return JSONResponse({"error": "Login required."}, status_code=401)
    if user.get("role") != "user":
        return JSONResponse({"error": "Only user accounts can upload images."}, status_code=403)

    contents = await file.read()
    img_preprocessed = preprocess_image(contents)

    predictions = MODEL.predict(img_preprocessed)
    score = np.max(predictions)
    class_idx = np.argmax(predictions)
    label = CLASS_NAMES[class_idx]

    fruit, status = label.split("_")
    confidence = f"{score * 100:.2f}%"

    extension = Path(file.filename or "upload.jpg").suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        extension = ".jpg"

    saved_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(6)}{extension}"
    saved_path = UPLOAD_DIR / saved_name
    saved_path.write_bytes(contents)
    image_url = f"/static/uploads/{saved_name}"

    result = {
        "fruit": fruit,
        "status": status,
        "confidence": confidence,
        "eatability": "Safe ✅" if status == "Fresh" else "Avoid ❌",
        "shelf_life": "5-7 Days" if status == "Fresh" else "Expired",
        "advice": "Keep in fridge." if status == "Fresh" else "Compost it.",
    }

    try:
        predictions_collection.insert_one(
            {
                "user_email": user.get("email", ""),
                "user_name": user.get("name", "User"),
                "fruit": fruit,
                "status": status,
                "confidence": confidence,
                "image_url": image_url,
                "created_at": datetime.now(timezone.utc),
            }
        )
    except PyMongoError as error:
        print(f"Could not save prediction record: {error}")

    return result


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if user.get("role") != "admin":
        return RedirectResponse("/", status_code=303)

    rows = []
    try:
        cursor = predictions_collection.find().sort("created_at", DESCENDING).limit(300)
        for item in cursor:
            created_at = item.get("created_at")
            if isinstance(created_at, datetime):
                created_at_text = created_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            else:
                created_at_text = "-"

            rows.append(
                {
                    "image_url": item.get("image_url", ""),
                    "user_email": item.get("user_email", "-"),
                    "fruit": item.get("fruit", "-"),
                    "status": item.get("status", "-"),
                    "confidence": item.get("confidence", "-"),
                    "created_at": created_at_text,
                }
            )
    except PyMongoError as error:
        print(f"Could not load admin data: {error}")

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "admin_name": user.get("name", "Admin"),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
