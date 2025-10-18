// Элементы DOM
const rangeInput = document.getElementById('rang');
const priceElement = document.getElementById('price-itog').querySelector('.price-text');
const probabilityElement = document.getElementById('ver');
const revenueElement = document.getElementById('inc');
const buttonCardSafe = document.getElementById('card1');
const buttonCardMean = document.getElementById('card2');
const buttonCardRisk = document.getElementById('card3'); // Исправлено: был card2


// Элементы для управления меню
const menuContainer = document.getElementById("menu-toggle-container");
const menuToggleBtn = document.getElementById('menu-toggle-btn');
const settingsMenu = document.getElementById('settings');
let isMenuVisible = true;

// Функция переключения меню
function toggleMenu() {
    isMenuVisible = !isMenuVisible;
    
    if (isMenuVisible) {
        // Показываем меню
        settingsMenu.classList.remove('collapsed');
        menuToggleBtn.classList.remove('collapsed');

        menuContainer.style.height = "380px";
    } else {
        // Скрываем меню
        settingsMenu.classList.add('collapsed');
        menuToggleBtn.classList.add('collapsed');


        menuContainer.style.height = "760px";
    }
}

// Добавляем обработчик события
menuToggleBtn.addEventListener('click', toggleMenu);

// Инициализируем состояние (можно начать со скрытого меню, если нужно)
// Для этого раскомментируйте следующие строки:
settingsMenu.classList.add('collapsed');
menuToggleBtn.classList.add('collapsed');
isMenuVisible = false;
menuContainer.style.height = "760px";

// Переменная для хранения данных
let priceCurve = [];

// Функция загрузки данных с бэкенда
async function loadPriceData() {
    try {
        const response = await fetch('/api/price-curve');
        const data = await response.json();
        priceCurve = data.price_curve;
        
        // Инициализируем слайдер после загрузки данных
        initializeSlider();
        
    } catch (error) {
        console.error('Ошибка загрузки данных:', error);
        // Fallback данные на случай ошибки
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
        const safeIndex = 0; // Первый элемент - минимальная цена
        rangeInput.value = safeIndex;
        updatePriceData(safeIndex);
        highlightActiveButton(this);
    });
    
    // Mean - средняя цена (максимальная доходность)
    buttonCardMean.addEventListener('click', function() {
        // Находим индекс с максимальной доходностью
        const meanIndex = findMaxRevenueIndex();
        rangeInput.value = meanIndex;
        updatePriceData(meanIndex);
        highlightActiveButton(this);
    });
    
    // Risk - максимальная цена (самый рискованный вариант)
    buttonCardRisk.addEventListener('click', function() {
        const riskIndex = priceCurve.length - 1; // Последний элемент - максимальная цена
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
    const selectedPrice = priceCurve[selectedIndex];
    
    // Обновляем интерфейс
    priceElement.textContent = `${Math.round(selectedPrice.price)} руб.`;
    probabilityElement.innerHTML = `Вероятность принятия: ${Math.round(selectedPrice.probability * 100)}%`;
    revenueElement.innerHTML = `Доходность: ${Math.round(selectedPrice.expected_revenue)} руб.`;
}

// Функция для подсветки активной кнопки
function highlightActiveButton(activeButton) {
    // Убираем подсветку со всех кнопок
    [buttonCardSafe, buttonCardMean, buttonCardRisk].forEach(button => {
        button.classList.remove('active');
    });
    
    // Добавляем подсветку активной кнопке
    activeButton.classList.add('active');
}

// Загружаем данные при загрузке страницы
document.addEventListener('DOMContentLoaded', loadPriceData);