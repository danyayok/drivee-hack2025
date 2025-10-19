// predict.js

// DOM элементы
const priceElement = document.getElementById('price-itog').querySelector('.price-text');
const probabilityElement = document.getElementById('ver');
const revenueElement = document.getElementById('inc');
const buttonSafe = document.getElementById('card1');
const buttonMean = document.getElementById('card2');
const buttonRisk = document.getElementById('card3');
const menuToggleBtn = document.getElementById('menu-toggle-btn');
const settingsMenu = document.getElementById('settings');
const menuContainer = document.getElementById('menu-toggle-container');
const rangeInput = document.getElementById('rang');

let currentOrder = null;
let priceCurve = [];
let selectedIndex = 0;

// -----------------------------
// Выбор случайного заказа при открытии меню
// -----------------------------
menuToggleBtn.addEventListener('click', async () => {
    console.log('Кнопка меню нажата');
    toggleMenu();

    if (!currentOrder || !menuVisible) {
        console.log('Загружаем новый заказ...');
        currentOrder = getRandomOrder();
        console.log('Выбран заказ:', currentOrder);
        await loadOptimalPrices(currentOrder);
        updateUI(selectedIndex);
    }
});

// -----------------------------
// Функция переключения меню
// -----------------------------
let menuVisible = false;
function toggleMenu() {
    menuVisible = !menuVisible;
    console.log('Меню видимо:', menuVisible);

    if (menuVisible) {
        settingsMenu.classList.remove('collapsed');
        menuToggleBtn.classList.remove('collapsed');
        menuContainer.style.backgroundImage = "url('/static/images/maps-2.png')";
        menuContainer.style.height = "380px";
    } else {
        settingsMenu.classList.add('collapsed');
        menuToggleBtn.classList.add('collapsed');
        menuContainer.style.backgroundImage = "url('/static/images/maps-1.png')";
        menuContainer.style.height = "760px";
    }
}

// -----------------------------
// Случайный заказ
// -----------------------------
function getRandomOrder() {
    const examples = [
        {distance_in_meters: 3500, duration_in_seconds: 600, pickup_in_meters: 500, pickup_in_seconds: 120, driver_rating: 4.8, user_rating: 4.9, price_start_local: 200, order_timestamp: "2024-01-15T12:00:00", driver_platform: "android", driver_reg_date: "2023-01-01", carname: "Toyota", carmodel: "Camry", driver_id: "29368889", user_id: "16458846"},
        {distance_in_meters: 2800, duration_in_seconds: 480, pickup_in_meters: 300, pickup_in_seconds: 90, driver_rating: 4.9, user_rating: 4.7, price_start_local: 180, order_timestamp: "2024-01-15T14:30:00", driver_platform: "ios", driver_reg_date: "2023-03-15", carname: "Honda", carmodel: "Civic", driver_id: "29368889", user_id: "16458846"},
        {distance_in_meters: 4200, duration_in_seconds: 720, pickup_in_meters: 600, pickup_in_seconds: 150, driver_rating: 4.7, user_rating: 4.8, price_start_local: 250, order_timestamp: "2024-01-15T16:45:00", driver_platform: "android", driver_reg_date: "2023-02-20", carname: "Hyundai", carmodel: "Solaris", driver_id: "29368889", user_id: "16458846"},
        {distance_in_meters: 1900, duration_in_seconds: 360, pickup_in_meters: 200, pickup_in_seconds: 60, driver_rating: 4.6, user_rating: 4.9, price_start_local: 150, order_timestamp: "2024-01-15T18:20:00", driver_platform: "ios", driver_reg_date: "2023-04-10", carname: "Kia", carmodel: "Rio", driver_id: "29368889", user_id: "16458846"},
        {distance_in_meters: 5100, duration_in_seconds: 840, pickup_in_meters: 700, pickup_in_seconds: 180, driver_rating: 4.8, user_rating: 4.6, price_start_local: 280, order_timestamp: "2024-01-15T20:15:00", driver_platform: "android", driver_reg_date: "2023-01-25", carname: "Volkswagen", carmodel: "Polo", driver_id: "29368889", user_id: "16458846"}
    ];
    return examples[Math.floor(Math.random() * examples.length)];
}

// -----------------------------
// Загрузка цен через Flask-прокси
// -----------------------------
async function loadOptimalPrices(order) {
    console.log('Начинаем загрузку оптимальных цен...');
    
    try {
        const response = await fetch('/proxy/get_optimal_prices', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(order)
        });
        
        console.log('Ответ получен, статус:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Данные от сервера:', data);

        // Сохраняем все поля и сортируем по цене
        priceCurve = data.price_curve
            .sort((a, b) => a.price - b.price)
            .map(p => ({
                price: p.price,
                probability: p.probability_percent,
                expected_revenue: p.expected_revenue,
                service_earnings: p.service_earnings,
                driver_earnings: p.driver_earnings
            }));

        console.log('Отсортированные данные priceCurve:', priceCurve);

    } catch (err) {
        console.error("Ошибка загрузки цен:", err);

        // fallback - тоже сортируем
        priceCurve = [
            {price: 205, probability: 67.1, expected_revenue: 137.5, service_earnings: 17.6, driver_earnings: 119.9},
            {price: 216, probability: 58.4, expected_revenue: 126.12, service_earnings: 16.14, driver_earnings: 109.97},
            {price: 183, probability: 70.7, expected_revenue: 129.35, service_earnings: 16.56, driver_earnings: 112.79},
            {price: 238, probability: 47, expected_revenue: 111.85, service_earnings: 14.32, driver_earnings: 97.53},
            {price: 162, probability: 76.2, expected_revenue: 123.51, service_earnings: 15.81, driver_earnings: 107.7}
        ].sort((a, b) => a.price - b.price);

        console.log('Fallback данные (отсортированные):', priceCurve);
    }
    
    // Инициализируем слайдер с отсортированными данными
    initSlider();
}

// -----------------------------
// Инициализация ползунка
// -----------------------------
function initSlider() {
    console.log('Инициализация слайдера, priceCurve.length:', priceCurve.length);
    
    if (priceCurve.length === 0) {
        console.error('Нет данных для инициализации слайдера');
        return;
    }

    rangeInput.min = 0;
    rangeInput.max = priceCurve.length - 1;
    rangeInput.step = 1;
    rangeInput.value = 0;
    selectedIndex = 0;

    console.log('Слайдер настроен:', {
        min: rangeInput.min,
        max: rangeInput.max,
        value: rangeInput.value,
        points: priceCurve.length
    });

    // Удаляем старые обработчики чтобы избежать дублирования
    rangeInput.oninput = null;
    buttonSafe.onclick = null;
    buttonMean.onclick = null;
    buttonRisk.onclick = null;

    // Добавляем новые обработчики
    rangeInput.addEventListener('input', () => {
        selectedIndex = parseInt(rangeInput.value);
        console.log('Слайдер изменен на индекс:', selectedIndex, 'Цена:', priceCurve[selectedIndex].price);
        updateUI(selectedIndex);
    });

    buttonSafe.addEventListener('click', () => {
        selectedIndex = 0;
        rangeInput.value = selectedIndex;
        console.log('Кнопка Safe:', selectedIndex, 'Цена:', priceCurve[selectedIndex].price);
        updateUI(selectedIndex);
    });

    buttonMean.addEventListener('click', () => {
        selectedIndex = findMaxRevenueIndex();
        rangeInput.value = selectedIndex;
        console.log('Кнопка Mean:', selectedIndex, 'Цена:', priceCurve[selectedIndex].price);
        updateUI(selectedIndex);
    });

    buttonRisk.addEventListener('click', () => {
        selectedIndex = priceCurve.length - 1;
        rangeInput.value = selectedIndex;
        console.log('Кнопка Risk:', selectedIndex, 'Цена:', priceCurve[selectedIndex].price);
        updateUI(selectedIndex);
    });

    // Обновляем интерфейс сразу после инициализации
    updateUI(selectedIndex);
}

// -----------------------------
// Найти индекс максимальной доходности
// -----------------------------
function findMaxRevenueIndex() {
    let maxIdx = 0;
    let maxRevenue = priceCurve[0].expected_revenue;
    for (let i = 1; i < priceCurve.length; i++) {
        if (priceCurve[i].expected_revenue > maxRevenue) {
            maxRevenue = priceCurve[i].expected_revenue;
            maxIdx = i;
        }
    }
    console.log('Максимальная доходность найдена по индексу:', maxIdx);
    return maxIdx;
}

// -----------------------------
// Обновление интерфейса
// -----------------------------
function updateUI(idx) {
    console.log('Обновление UI, индекс:', idx);
    
    if (!priceCurve || priceCurve.length === 0) {
        console.error('Нет данных для обновления интерфейса');
        priceElement.textContent = 'Нет данных';
        probabilityElement.textContent = 'Вероятность принятия: —';
        revenueElement.textContent = 'Доходность: —';
        return;
    }

    if (idx < 0 || idx >= priceCurve.length) {
        console.error('Некорректный индекс:', idx);
        return;
    }

    const selected = priceCurve[idx];
    console.log('Выбранные данные для отображения:', selected);

    // Обновляем цену и доходность
    priceElement.textContent = `${selected.price} руб.`;
    probabilityElement.textContent = `Вероятность принятия: ${selected.probability.toFixed(1)}%`;
    revenueElement.textContent = `Доходность: ${selected.expected_revenue.toFixed(1)} руб.`;

    // Обновляем данные заказа
    if (currentOrder) {
        document.querySelector('#zagalovok h1').textContent = `Заказ №${Math.floor(Math.random() * 1000) + 1000}`;
        document.querySelector('#first-row .first-text-pred').textContent = `Расстояние: ${(currentOrder.distance_in_meters / 1000).toFixed(1)} км`;
        document.querySelector('#first-row .second-text-pred').textContent = `Машина: ${currentOrder.carname || '—'} ${currentOrder.carmodel || ''}`;
        document.querySelector('#second-row .first-text-pred').textContent = `Водитель: ${currentOrder.driver_id || '—'}`;
        document.querySelector('#second-row .second-text-pred').textContent = `Рейтинг: ${currentOrder.driver_rating || '—'}`;
    }

    console.log('Интерфейс успешно обновлен');
}

// -----------------------------
// Отправка заказа
// -----------------------------
document.getElementById('button').addEventListener('click', async () => {
    console.log('Кнопка отправки нажата');
    
    if (!currentOrder || priceCurve.length === 0) {
        alert('Нет данных заказа!');
        return;
    }

    const selected = priceCurve[selectedIndex];
    const orderToSend = {
        ...currentOrder,
        price_bid_local: selected.price
    };

    console.log('Отправляемые данные:', orderToSend);

    try {
        const resp = await fetch('/proxy/record_order', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(orderToSend)
        });
        const result = await resp.json();
        alert('Заказ отправлен!');
        console.log('Заказ отправлен:', result);
    } catch (err) {
        console.error('Ошибка отправки заказа:', err);
        alert('Ошибка отправки заказа!');
    }
});

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    console.log('Страница загружена, DOM готов');
    // Начальное состояние - меню скрыто
    settingsMenu.classList.add('collapsed');
    menuToggleBtn.classList.add('collapsed');
    menuContainer.style.backgroundImage = "url('/static/images/maps-1.png')";
    menuContainer.style.height = "760px";
});