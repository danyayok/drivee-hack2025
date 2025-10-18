"""
Реальные тестовые данные из предоставленного датасета
"""

def get_real_test_samples():
    """Возвращает реальные примеры заказов для тестирования"""
    return [
        {
            "order_id": "7841348880",
            "order_timestamp": "2020-05-01T00:05:14",
            "distance_in_meters": 3404.0,
            "duration_in_seconds": 486.0,
            "pickup_in_meters": 790.0,
            "pickup_in_seconds": 169.0,
            "driver_rating": 5.0,
            "user_rating": 4.8,
            "price_start_local": 180.0,
            "price_bid_local": 200.0,
            "driver_platform": "Android",  # Исправлено
            "driver_reg_date": "2019-09-22",
            "carmodel": "Logan",
            "carname": "Renault",
            "is_done": "done"
        },
        {
            "order_id": "7841349235",
            "order_timestamp": "2020-05-01T00:06:25",
            "distance_in_meters": 994.0,
            "duration_in_seconds": 176.0,
            "pickup_in_meters": 469.0,
            "pickup_in_seconds": 85.0,
            "driver_rating": 5.0,
            "user_rating": 4.9,
            "price_start_local": 160.0,
            "price_bid_local": 200.0,
            "driver_platform": "Android",  # Исправлено
            "driver_reg_date": "2019-02-04",
            "carmodel": "Sandero Stepway",
            "carname": "Renault",
            "is_done": "done"
        },
        {
            "order_id": "7841349809",
            "order_timestamp": "2020-05-01T00:08:19",
            "distance_in_meters": 2749.0,
            "duration_in_seconds": 471.0,
            "pickup_in_meters": 436.0,
            "pickup_in_seconds": 160.0,
            "driver_rating": 5.0,
            "user_rating": 4.7,
            "price_start_local": 150.0,
            "price_bid_local": 200.0,
            "driver_platform": "Android",  # Исправлено
            "driver_reg_date": "2017-12-21",
            "carmodel": "Avensis",
            "carname": "Toyota",
            "is_done": "done"
        },
        {
            "order_id": "7841351242",
            "order_timestamp": "2020-05-01T00:13:15",
            "distance_in_meters": 2339.0,
            "duration_in_seconds": 323.0,
            "pickup_in_meters": 976.0,
            "pickup_in_seconds": 143.0,
            "driver_rating": 4.94737,
            "user_rating": 4.6,
            "price_start_local": 250.0,
            "price_bid_local": 250.0,
            "driver_platform": "iOS",
            "driver_reg_date": "2017-12-23",
            "carmodel": "Logan",
            "carname": "Renault",
            "is_done": "done"
        }
    ]