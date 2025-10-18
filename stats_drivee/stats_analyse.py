import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def recalculate_ml_profit_realistic():
    """
    Пересчет прибыли от ML с динамическим расчетом acceptance rate
    """
    # Реальные данные
    avg_daily_orders = 1613       # выполненные заказы
    avg_daily_revenue = 52350     # выручка в день
    avg_order_value = 216         # средний чек
    active_drivers = 102          # активные водители

    print("🔄 ПЕРЕСЧЕТ ПРИБЫЛИ ОТ ML С ДИНАМИЧЕСКИМ ACCEPTANCE RATE")
    print("=" * 60)

    # ML-параметры
    ml_coverage = 0.4             # 40% поездок используют ML
    price_increase = 0.12         # +12% к цене
    acceptance_boost = 0.08       # +8% к acceptance rate

    # Расчет базового acceptance rate
    avg_daily_requests = avg_daily_orders / 0.6  # предположим исходный acceptance_rate = 60%
    base_acceptance_rate = avg_daily_orders / avg_daily_requests
    print(f"📊 Базовый acceptance rate: {base_acceptance_rate:.1%}")

    # Заказы с ML
    orders_with_ml = avg_daily_orders * ml_coverage
    revenue_increase_per_order = avg_order_value * price_increase
    ml_profit_from_prices = orders_with_ml * revenue_increase_per_order * 0.15

    # Дополнительные заказы из-за ML
    additional_orders = avg_daily_requests * ml_coverage * acceptance_boost
    ml_profit_from_orders = additional_orders * avg_order_value * 0.15

    total_ml_profit = ml_profit_from_prices + ml_profit_from_orders

    # Новый acceptance rate после ML
    accepted_orders_after_ml = avg_daily_orders + additional_orders
    new_acceptance_rate = accepted_orders_after_ml / avg_daily_requests

    print(f"\n🎯 Эффект от ML:")
    print(f"• Охват ML: {ml_coverage:.0%} поездок")
    print(f"• Повышение цены: +{price_increase:.0%}")
    print(f"• Прирост acceptance rate: +{acceptance_boost:.0%}")
    print(f"• Новая вероятность принятия заказа: {new_acceptance_rate:.1%}")
    print(f"• Прибыль от повышения цен: {ml_profit_from_prices:,.0f} руб/день")
    print(f"• Прибыль от доп. заказов: {ml_profit_from_orders:,.0f} руб/день")
    print(f"• Общая прибыль от ML: {total_ml_profit:,.0f} руб/день")

    # Расчет ROI
    investment = 75000
    monthly_costs = 21500
    daily_costs = monthly_costs / 30
    net_daily_profit = total_ml_profit - daily_costs

    break_even_days = investment / net_daily_profit if net_daily_profit > 0 else float('inf')
    annual_ml_profit = net_daily_profit * 365
    total_investment = investment + monthly_costs * 12
    roi_percentage = (annual_ml_profit / total_investment) * 100

    growth_percentage = (total_ml_profit / avg_daily_revenue) * 100

    # Эффект для водителей
    driver_orders_before = avg_daily_orders / active_drivers
    driver_earnings_before = (avg_order_value * base_acceptance_rate * 0.85) * driver_orders_before

    driver_orders_after = accepted_orders_after_ml / active_drivers
    driver_earnings_after = (avg_order_value * (1 + price_increase) * 0.85) * driver_orders_after

    driver_earnings_growth = (driver_earnings_after / driver_earnings_before - 1) * 100

    print(f"\n🚗 Эффект для водителей:")
    print(f"• Поездок на водителя: {driver_orders_before:.1f} → {driver_orders_after:.1f}")
    print(f"• Заработок водителя: {driver_earnings_before:,.0f} → {driver_earnings_after:,.0f} руб/день")
    print(f"• Рост дохода: +{driver_earnings_growth:.1f}%")

    return {
        'avg_daily_orders': avg_daily_orders,
        'avg_order_value': avg_order_value,
        'active_drivers': active_drivers,
        'current_daily_revenue': avg_daily_revenue,
        'ml_daily_profit': total_ml_profit,
        'net_daily_profit': net_daily_profit,
        'growth_percentage': growth_percentage,
        'break_even_days': break_even_days,
        'annual_ml_profit': annual_ml_profit,
        'roi_percentage': roi_percentage,
        'driver_earnings_before': driver_earnings_before,
        'driver_earnings_after': driver_earnings_after,
        'driver_earnings_growth': driver_earnings_growth,
        'acceptance_rate': new_acceptance_rate
    }



import matplotlib.pyplot as plt
import seaborn as sns

def create_drivee_presentation_charts(metrics):
    """
    Создает графики для презентации по прибыли ML-системы Drivee.
    Сохраняет все графики в PNG.
    """
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # --- Слайд 2: Базовые показатели ---
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ['Поездки/день', 'Средний чек', 'Активные водители', 'Acceptance rate']
    values = [metrics['avg_daily_orders'], metrics['avg_order_value'],
              metrics['active_drivers'], metrics['acceptance_rate']*100]
    bars = ax.bar(categories, values, color=['#51cf66', '#339af0', '#fcc419', '#ff6b6b'])
    ax.set_title('Базовые показатели', fontsize=16, fontweight='bold')
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{value:.0f}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('slide_2_basics.png', dpi=300)
    plt.close()

    # --- Слайд 3: Эффект от ML ---
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ['До ML', 'После ML']
    revenue_values = [metrics['current_daily_revenue'],
                      metrics['current_daily_revenue'] + metrics['ml_daily_profit']]
    bars = ax.bar(categories, revenue_values, color=['#ff6b6b', '#51cf66'])
    ax.set_title('Эффект от ML', fontsize=16, fontweight='bold')
    for bar, value in zip(bars, revenue_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{value:,.0f} ₽', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('slide_3_ml_effect.png', dpi=300)
    plt.close()

    # --- Слайд 4: Накопленная прибыль и окупаемость ---
    months = list(range(0, 13))
    cumulative_profit = [metrics['net_daily_profit'] * 30 * m for m in months]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(months, cumulative_profit, linewidth=3, marker='o', color='#20c997')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    break_even_month = metrics['break_even_days'] / 30
    if break_even_month <= 12:
        ax.axvline(x=break_even_month, color='orange', linestyle='--')
        ax.text(break_even_month, max(cumulative_profit)/2,
                f'Окупаемость\n{break_even_month:.1f} мес', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    ax.set_title('Накопленная прибыль и окупаемость', fontsize=16, fontweight='bold')
    ax.set_xlabel('Месяцы')
    ax.set_ylabel('Накопленная прибыль (₽)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('slide_4_roi.png', dpi=300)
    plt.close()

    # --- Слайд 5: Эффект для водителей ---
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ['До ML', 'После ML']
    driver_income = [metrics['driver_earnings_before'], metrics['driver_earnings_after']]
    bars = ax.bar(categories, driver_income, color=['#ff922b', '#339af0'])
    ax.set_title('Эффект для водителей', fontsize=16, fontweight='bold')
    for bar, value in zip(bars, driver_income):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{value:,.0f} ₽', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('slide_5_drivers.png', dpi=300)
    plt.close()

    print("✅ Графики для презентации созданы и сохранены в текущей папке.")



# Запуск пересчета
if __name__ == "__main__":
    print("🔄 ЗАПУСК ПЕРЕСЧЕТА С РЕАЛИСТИЧНЫМИ ПАРАМЕТРАМИ...")
    metrics = recalculate_ml_profit_realistic()
    metrics.update({
        'avg_daily_orders': 1613,
        'avg_order_value': 216,
        'active_drivers': 102,
        'acceptance_rate': 0.7,
        'driver_earnings_before': (216*0.85)*(1613/102),
        'driver_earnings_after': (216*1.12*0.85)*(1613*(1+0.4*0.08)/102)
    })
    create_drivee_presentation_charts(metrics)


    print("\n" + "=" * 70)
    print("🎯 ИТОГОВОЕ РЕЗЮМЕ (РЕАЛИСТИЧНЫЙ СЦЕНАРИЙ)")
    print("=" * 70)
    print(f"📈 Текущая дневная выручка: {metrics['current_daily_revenue']:,.0f} руб")
    print(f"🚀 Прибыль от ML в день: {metrics['ml_daily_profit']:,.0f} руб")
    print(f"💸 Чистая прибыль в день: {metrics['net_daily_profit']:,.0f} руб")
    print(f"📊 Увеличение выручки: +{metrics['growth_percentage']:.1f}%")
    print(f"⏱️ Окупаемость: {metrics['break_even_days']:.1f} дней")
    print(f"💰 Годовая прибыль от ML: {metrics['annual_ml_profit']:,.0f} руб")
    print(f"🎯 ROI: {metrics['roi_percentage']:.0f}%")
    print(f"🚗 Рост доходов водителей: +{metrics['driver_earnings_growth']:.1f}%")
    print("=" * 70)