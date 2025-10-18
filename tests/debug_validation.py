from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Тестовые данные с правильным регистром
sample_data = {
    "distance_in_meters": 3404.0,
    "duration_in_seconds": 486.0,
    "pickup_in_meters": 790.0,
    "pickup_in_seconds": 169.0,
    "driver_rating": 5.0,
    "user_rating": 4.8,
    "price_start_local": 180.0,
    "order_timestamp": "2020-05-01T00:05:14",
    "driver_platform": "Android",  # Исправлено на правильный регистр
    "driver_reg_date": "2019-09-22",
    "carmodel": "Logan",
    "carname": "Renault"
}

print("Testing with data:")
print(sample_data)
print()

response = client.post("/api/v1/get_optimal_prices", json=sample_data)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

if response.status_code == 200:
    data = response.json()
    print(f"Success! Got {len(data.get('price_curve', []))} price suggestions")