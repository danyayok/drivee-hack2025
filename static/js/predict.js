// Элементы DOM
const rangeInput = document.getElementById('rang');
const priceElement = document.getElementById('price-itog').querySelector('.price-text');
const probabilityElement = document.getElementById('ver');
const revenueElement = document.getElementById('inc');
const buttonCardSafe = document.getElementById('card1');
const buttonCardMean = document.getElementById('card2');
const buttonCardRisk = document.getElementById('card3');

// Элементы для управления меню
const menuContainer = document.getElementById("menu-toggle-container");
const menuToggleBtn = document.getElementById('menu-toggle-btn');
const settingsMenu = document.getElementById('settings');
let isMenuVisible = true;

// Захардкоженные примеры данных для отправки
const orderExamples = [
    {
        "distance_in_meters": 3500.0,
        "duration_in_seconds": 600.0,
        "pickup_in_meters": 500.0,
        "pickup_in_seconds": 120.0,
        "driver_rating": 4.8,
        "user_rating": 4.9,
        "price_start_local": 200.0,
        "order_timestamp": "2024-01-15T12:00:00",
        "driver_platform": "android",
        "driver_reg_date": "2023-01-01",
        "carname": "Toyota",
        "carmodel": "Camry",
        "driver_id": "29368889",
        "user_id": "16458846"
    },
    {
        "distance_in_meters": 2800.0,
        "duration_in_seconds": 480.0,
        "pickup_in_meters": 300.0,
        "pickup_in_seconds": 90.0,
        "driver_rating": 4.9,
        "user_rating": 4.7,
        "price_start_local": 180.0,
        "order_timestamp": "2024-01-15T14:30:00",
        "driver_platform": "ios",
        "driver_reg_date": "2023-03-15",
        "carname": "Honda",
        "carmodel": "Civic",
        "driver_id": "29368889",
        "user_id": "16458846"
    },
    {
        "distance_in_meters": 4200.0,
        "duration_in_seconds": 720.0,
        "pickup_in_meters": 600.0,
        "pickup_in_seconds": 150.0,
        "driver_rating": 4.7,
        "user_rating": 4.8,
        "price_start_local": 250.0,
        "order_timestamp": "2024-01-15T16:45:00",
        "driver_platform": "android",
        "driver_reg_date": "2023-02-20",
        "carname": "Hyundai",
        "carmodel": "Solaris",
        "driver_id": "29368889",
        "user_id": "16458846"
    },
    {
        "distance_in_meters": 1900.0,
        "duration_in_seconds": 360.0,
        "pickup_in_meters": 200.0,
        "pickup_in_seconds": 60.0,
        "driver_rating": 4.6,
        "user_rating": 4.9,
        "price_start_local": 150.0,
        "order_timestamp": "2024-01-15T18:20:00",
        "driver_platform": "ios",
        "driver_reg_date": "2023-04-10",
        "carname": "Kia",
        "carmodel": "Rio",
        "driver_id": "29368889",
        "user_id": "16458846"
    },
    {
        "distance_in_meters": 5100.0,
        "duration_in_seconds": 840.0,
        "pickup_in_meters": 700.0,
        "pickup_in_seconds": 180.0,
        "driver_rating": 4.8,
        "user_rating": 4.6,
        "price_start_local": 280.0,
        "order_timestamp": "2024-01-15T20:15:00",
        "driver_platform": "android",
        "driver_reg_date": "2023-01-25",
        "carname": "Volkswagen",
        "carmodel": "Polo",
        "driver_id": "29368889",
        "user_id": "16458846"
    }
];

// Текущие данные
let currentOrderData = null;
let priceCurve = [];
let selectedPrice = null;

// Функция переключения меню
function toggleMenu() {
    isMenuVisible = !isMenuVisible;
    
    if (isMenuVisible) {
        // Показываем меню - выбираем случайный заказ и загружаем данные
        settingsMenu.classList.remove('collapsed');
        menuToggleBtn.classList.remove('collapsed');
        menuContainer.style.backgroundImage = "url('/static/images/maps-2.png')";
        menuContainer.style.height = "380px";
        
        // Выбираем случайный заказ и загружаем данные
        selectRandomOrderAndLoadPrices();
    } else {
        // Скрываем меню
        settingsMenu.classList.add('collapsed');
        menuToggleBtn.classList.add('collapsed');
        menuContainer.style.backgroundImage = "url('/static/images/maps-1.png')";
        menuContainer.style.height = "760px";
    }
}

// Функция выбора случайного заказа и загрузки цен
async function selectRandomOrderAndLoadPrices() {
    try {
        // Выбираем случайный заказ
        const randomIndex = Math.floor(Math.random() * orderExamples.length);
        currentOrderData = orderExamples[randomIndex];
        
        // Обновляем информацию о заказе в интерфейсе
        updateOrderInfo(currentOrderData);
        
        // Загружаем оптимальные цены с сервера
        await loadOptimalPrices(currentOrderData);
        
    } catch (error) {
        console.error('Ошибка при выборе заказа:', error);
        // Fallback на локальные данные
        priceCurve = [
            {"price": 315.0, "probability": 0.92, "expected_revenue": 289.8},
            {"price": 350.0, "probability": 0.85, "expected_revenue": 297.5},
            {"price": 420.0, "probability": 0.72, "expected_revenue": 302.4},
            {"price": 490.0, "probability": 0.55, "expected_revenue": 269.5},
            {"price": 525.0, "probability": 0.45, "expected_revenue": 236.25}
        ];
        initializeSlider();
    }
}

// Функция обновления информации о заказе
function updateOrderInfo(orderData) {
    const orderNumberElement = document.querySelector('#zagalovok .predict-text');
    const fromElement = document.querySelector('#first-row .first-text-pred');
    const toElement = document.querySelector('#second-row .first-text-pred');
    const proposedPriceElement = document.querySelector('#first-row .second-text-pred');
    const timeElement = document.querySelector('#second-row .second-text-pred');
    
    // Генерируем случайный номер заказа
    const orderNumber = Math.floor(Math.random() * 10000) + 1000;
    
    // Обновляем интерфейс
    orderNumberElement.textContent = `Заказ №${orderNumber}`;
    fromElement.textContent = `Расстояние: ${Math.round(orderData.distance_in_meters / 1000)} км`;
    toElement.textContent = `Машина: ${orderData.carmodel}`;
    proposedPriceElement.textContent = `Предложенная цена: ${Math.round(orderData.price_start_local)} руб.`;
    
    // Форматируем время
    const orderTime = new Date(orderData.order_timestamp);
    const timeString = orderTime.toLocaleTimeString('ru-RU', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    timeElement.textContent = `Время подачи: ${timeString}`;
}

// Функция загрузки оптимальных цен с сервера
async function loadOptimalPrices(orderData) {
    try {
        const response = await fetch('http://5.129.193.114:8000/api/v1/get_optimal_prices', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(orderData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Преобразуем данные в нужный формат
        priceCurve = data.price_curve.map(item => ({
            price: item.price,
            probability: item.probability_percent / 100,
            expected_revenue: item.expected_revenue
        }));
        
        // Инициализируем слайдер с новыми данными
        initializeSlider();
        
    } catch (error) {
        console.error('Ошибка загрузки оптимальных цен:', error);
        // Fallback данные
        priceCurve = [
            {"price": 315.0, "probability": 0.92, "expected_revenue": 289.8},
            {"price": 350.0, "probability": 0.85, "expected_revenue": 297.5},
            {"price": 420.0, "probability": 0.72, "expected_revenue": 302.4},
            {"price": 490.0, "probability": 0.55, "expected_revenue": 269.5},
            {"price": 525.0, "probability": 0.45, "expected_revenue": 236.25}
        ];
        initializeSlider();
    }
}

// Функция отправки выбранного заказа
async function submitOrder() {
    if (!currentOrderData || !selectedPrice) {
        alert('Пожалуйста, выберите цену перед отправкой');
        return;
    }
    
    try {
        const orderData = {
            order_id: Math.floor(Math.random() * 1000000).toString(),
            order_timestamp: currentOrderData.order_timestamp,
            distance_in_meters: currentOrderData.distance_in_meters,
            duration_in_seconds: currentOrderData.duration_in_seconds,
            tender_id: `tender_${Math.floor(Math.random() * 100000)}`,
            tender_timestamp: new Date().toISOString(),
            driver_id: parseInt(currentOrderData.driver_id),
            driver_reg_date: currentOrderData.driver_reg_date,
            driver_rating: currentOrderData.driver_rating,
            carmodel: currentOrderData.carmodel,
            carname: currentOrderData.carname,
            platform: currentOrderData.driver_platform,
            pickup_in_meters: currentOrderData.pickup_in_meters,
            pickup_in_seconds: currentOrderData.pickup_in_seconds,
            user_id: parseInt(currentOrderData.user_id),
            price_start_local: currentOrderData.price_start_local,
            price_bid_local: selectedPrice.price,
            is_done: true,
            user_rating: currentOrderData.user_rating
        };
        
        const response = await fetch('http://5.129.193.114:8000/api/v1/record_order', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(orderData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Заказ успешно отправлен:', result);
        alert('Заказ успешно отправлен!');
        
    } catch (error) {
        console.error('Ошибка отправки заказа:', error);
        alert('Ошибка отправки заказа. Попробуйте еще раз.');
    }
}

// Функция инициализации слайдера
function initializeSlider() {
    // Настраиваем слайдер по количеству точек
    rangeInput.min = 0;
    rangeInput.max = priceCurve.length - 1;
    rangeInput.value = 0;
    
    // Показываем первый вариант
    updatePriceData(0);
    
    // Добавляем обработчик слайдера
    rangeInput.addEventListener('input', function() {
        const selectedIndex = parseInt(this.value);
        updatePriceData(selectedIndex);
    });
    
    // Добавляем обработчики для кнопок
    addButtonHandlers();
}

// Функция добавления обработчиков для кнопок
function addButtonHandlers() {
    // Safe - минимальная цена (самый безопасный вариант)
    buttonCardSafe.addEventListener('click', function() {
        const safeIndex = 0;
        rangeInput.value = safeIndex;
        updatePriceData(safeIndex);
        highlightActiveButton(this);
    });
    
    // Mean - средняя цена (максимальная доходность)
    buttonCardMean.addEventListener('click', function() {
        const meanIndex = findMaxRevenueIndex();
        rangeInput.value = meanIndex;
        updatePriceData(meanIndex);
        highlightActiveButton(this);
    });
    
    // Risk - максимальная цена (самый рискованный вариант)
    buttonCardRisk.addEventListener('click', function() {
        const riskIndex = priceCurve.length - 1;
        rangeInput.value = riskIndex;
        updatePriceData(riskIndex);
        highlightActiveButton(this);
    });
}

// Функция для нахождения индекса с максимальной доходностью
function findMaxRevenueIndex() {
    let maxRevenue = -1;
    let maxIndex = 0;
    
    for (let i = 0; i < priceCurve.length; i++) {
        if (priceCurve[i].expected_revenue > maxRevenue) {
            maxRevenue = priceCurve[i].expected_revenue;
            maxIndex = i;
        }
    }
    
    return maxIndex;
}

// Функция обновления данных
function updatePriceData(selectedIndex) {
    selectedPrice = priceCurve[selectedIndex];
    
    // Обновляем интерфейс
    priceElement.textContent = `${Math.round(selectedPrice.price)} руб.`;
    probabilityElement.innerHTML = `Вероятность принятия: ${Math.round(selectedPrice.probability * 100)}%`;
    revenueElement.innerHTML = `Доходность: ${Math.round(selectedPrice.expected_revenue)} руб.`;
}

// Функция для подсветки активной кнопки
function highlightActiveButton(activeButton) {
    [buttonCardSafe, buttonCardMean, buttonCardRisk].forEach(button => {
        button.classList.remove('active');
    });
    activeButton.classList.add('active');
}

// Добавляем обработчик для кнопки отправки
document.getElementById('button').addEventListener('click', submitOrder);

// Добавляем обработчик события для кнопки меню
menuToggleBtn.addEventListener('click', toggleMenu);

// Инициализируем состояние
settingsMenu.classList.add('collapsed');
menuToggleBtn.classList.add('collapsed');
isMenuVisible = false;
menuContainer.style.height = "760px";
menuContainer.style.backgroundImage = "url('/static/images/maps-1.png')";

// Загружаем начальные данные при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    // Можно сразу выбрать случайный заказ или оставить пустым до открытия меню
});