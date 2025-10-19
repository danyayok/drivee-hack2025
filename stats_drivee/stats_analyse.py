"""
drivee_presentation_generation.py

Скрипт генерирует ВСЕ данные и артефакты, необходимые для презентации: 
- Описывает рубрику оценки (баллы 0-100 по 5 категориям),
- Рассчитывает финансовые сценарии (полные расходы, операционные, разовые, зарплаты),
- Считает окупаемость, ROI и годовую прибыль для двух сценариев: штатный middle-специалист и частично занятый за 20-30k.
- Загружает результаты раннего анализа (price_grid_predictions.csv и графики) если они есть и использует реальные метрики из датасета
- Сохраняет итоговые таблицы и простую презентацию (pptx) с ключевыми слайдами.

Как запускать:
    python drivee_presentation_generation.py --data /mnt/data/train.csv --price_grid /mnt/data/drivee_plots/price_grid_predictions.csv

Файлы вывода (по умолчанию):
    outputs/summary.csv
    outputs/costs_and_roi.csv
    outputs/presentation.pptx
    outputs/plots/acceptance_vs_price.png (если найдено)
    outputs/plots/expected_rev_vs_price.png (если найдено)

Автор: ChatGPT
"""

import argparse
import os
import pandas as pd
import numpy as np
from datetime import timedelta
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor


# -------------------------- Helper functions --------------------------

def safe_mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def format_rub(v):
    return f"{v:,.0f} ₽"


# -------------------------- Financial logic --------------------------

def compute_roi_and_payback(investment_one_time, monthly_costs, daily_ml_profit, margin_platform=0.15):
    """Возвращает словарь: net_daily_profit, break_even_days, annual_ml_profit, roi_percent
    roi = annual_ml_profit / total_investment
    total_investment = investment_one_time + 12*monthly_costs
    """
    daily_costs = monthly_costs / 30.0
    net_daily_profit = daily_ml_profit - daily_costs
    if net_daily_profit <= 0:
        break_even_days = float('inf')
    else:
        break_even_days = investment_one_time / net_daily_profit
    annual_ml_profit = net_daily_profit * 365
    total_investment = investment_one_time + 12 * monthly_costs
    roi_percentage = (annual_ml_profit / total_investment) * 100 if total_investment > 0 else float('inf')

    return {
        'daily_costs': daily_costs,
        'net_daily_profit': net_daily_profit,
        'break_even_days': break_even_days,
        'annual_ml_profit': annual_ml_profit,
        'total_investment': total_investment,
        'roi_percentage': roi_percentage
    }


# -------------------------- Presentation rubric --------------------------

def get_rubric():
    rubric = [
        { 'name': 'Корректность модели', 'max_score': 30, 'criteria': [
            'Адекватность методики рынку',
            'Использование статистики/ML',
            'Проверки стабильности и доверительные интервалы'
        ]},
        { 'name': 'Применимость и практичность', 'max_score': 20, 'criteria': [
            'Сложность интеграции',
            'Юзабилити для водителей',
            'Требуемая инфраструктура'
        ]},
        { 'name': 'Качество визуализации', 'max_score': 20, 'criteria': [
            'Графики и схемы понятны',
            'Наглядность для неспециалистов'
        ]},
        { 'name': 'Инновационность подхода', 'max_score': 20, 'criteria': [
            'Оригинальные методы',
            'Нестандартные источники данных или идеи'
        ]},
        { 'name': 'Презентация решения', 'max_score': 10, 'criteria': [
            'Ясность объяснения',
            'Аргументация выбора методики'
        ]}
    ]
    return rubric


# -------------------------- Main generation --------------------------

def generate_presentation(data_path=None, price_grid_path=None, outputs_dir='outputs', use_provided_summary=None):
    safe_mkdir(outputs_dir)
    safe_mkdir(os.path.join(outputs_dir, 'plots'))

    # ---------------- Load data if exists ----------------
    df = None
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # minimal preprocessing
        for col in ['order_timestamp','tender_timestamp','driver_reg_date']:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    pass
        # compute actual revenue and daily
        if 'price_bid_local' in df.columns:
            df['price_bid_local'] = pd.to_numeric(df['price_bid_local'], errors='coerce')
            df['is_done'] = df['is_done'].astype(str)
            df['accepted'] = df['is_done'].str.lower().eq('done').astype(int)
            actual_revenue = df.loc[df['accepted']==1, 'price_bid_local'].sum()
            period_days = (df['order_timestamp'].max() - df['order_timestamp'].min()).days + 1
            if period_days <= 0:
                period_days = 1
            actual_revenue_daily = actual_revenue / period_days
        else:
            actual_revenue = np.nan
            actual_revenue_daily = np.nan
    else:
        actual_revenue = np.nan
        actual_revenue_daily = np.nan

    # ---------------- Load price grid predictions (optional) ----------------
    price_grid = None
    if price_grid_path and os.path.exists(price_grid_path):
        price_grid = pd.read_csv(price_grid_path)

    # ---------------- Default / provided summary numbers (как в твоём сообщении) ----------------
    # allow override if use_provided_summary dict is given
    defaults = {
        'current_daily_revenue': 52350.0,   # как в шаблоне
        'ml_profit_daily': 4181.0,
        'net_daily_profit': 798.0,
        'revenue_uplift_pct': 8.0,
        'payback_days': 94.0,
        'annual_profit_ml': 291110.0,
        'roi_pct': 23.0,
        'driver_income_growth_pct': 15.6
    }
    if use_provided_summary and isinstance(use_provided_summary, dict):
        defaults.update(use_provided_summary)

    # If we computed actual_revenue_daily from data, prefer it
    if not np.isnan(actual_revenue_daily):
        defaults['current_daily_revenue'] = actual_revenue_daily

    # ---------------- Cost assumptions ----------------
    # Server tiers from user message and common sense
    servers = {
        'small': { 'monthly': 1500, 'daily': 70, 'annual': 18000 },
        'large': { 'monthly': 100000, 'daily': 3225, 'annual': 1200000 }
    }
    dev_one_time_range = (50000, 100000)  # единовременно
    middle_salary = 100000  # месяц
    part_time_low = 20000
    part_time_high = 30000

    # ---------------- Scenario A: Hire Middle specialist (штатный) ----------------
    # Costs: one-time dev+infra + monthly salary + server medium (choose small or large?)
    invest_one_time = 75000  # предположение
    monthly_costs_a = servers['small']['monthly'] + middle_salary  # сервер + зарплата
    daily_ml_profit = defaults['ml_profit_daily']

    a_res = compute_roi_and_payback(invest_one_time, monthly_costs_a, daily_ml_profit)

    # ---------------- Scenario B: Part-time specialist ----------------
    invest_one_time_b = 50000
    monthly_costs_b_low = servers['small']['monthly'] + part_time_low
    monthly_costs_b_high = servers['small']['monthly'] + part_time_high

    b_res_low = compute_roi_and_payback(invest_one_time_b, monthly_costs_b_low, daily_ml_profit)
    b_res_high = compute_roi_and_payback(invest_one_time_b, monthly_costs_b_high, daily_ml_profit)

    # ---------------- Aggregate outputs ----------------
    summary_rows = []

    summary_rows.append({
        'scenario': 'baseline_from_data' if not np.isnan(actual_revenue_daily) else 'baseline_provided',
        'current_daily_revenue': defaults['current_daily_revenue'],
        'ml_daily_profit': daily_ml_profit,
        'net_daily_profit': a_res['net_daily_profit'],
        'revenue_uplift_pct': defaults['revenue_uplift_pct'],
        'payback_days_estimate': a_res['break_even_days'],
        'annual_ml_profit': a_res['annual_ml_profit'],
        'roi_pct': a_res['roi_percentage'],
        'note': 'Staffed middle specialist, small server'
    })

    summary_rows.append({
        'scenario': 'part_time_low',
        'current_daily_revenue': defaults['current_daily_revenue'],
        'ml_daily_profit': daily_ml_profit,
        'net_daily_profit': b_res_low['net_daily_profit'],
        'revenue_uplift_pct': defaults['revenue_uplift_pct'],
        'payback_days_estimate': b_res_low['break_even_days'],
        'annual_ml_profit': b_res_low['annual_ml_profit'],
        'roi_pct': b_res_low['roi_percentage'],
        'note': 'Part-time 20k, small server'
    })

    summary_rows.append({
        'scenario': 'part_time_high',
        'current_daily_revenue': defaults['current_daily_revenue'],
        'ml_daily_profit': daily_ml_profit,
        'net_daily_profit': b_res_high['net_daily_profit'],
        'revenue_uplift_pct': defaults['revenue_uplift_pct'],
        'payback_days_estimate': b_res_high['break_even_days'],
        'annual_ml_profit': b_res_high['annual_ml_profit'],
        'roi_pct': b_res_high['roi_percentage'],
        'note': 'Part-time 30k, small server'
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(outputs_dir, 'costs_and_roi.csv'), index=False)

    # ---------------- Save rubric and explanations ----------------
    rubric = get_rubric()
    rubric_df_rows = []
    for r in rubric:
        rubric_df_rows.append({ 'category': r['name'], 'max_score': r['max_score'], 'criteria': ' | '.join(r['criteria']) })
    rubric_df = pd.DataFrame(rubric_df_rows)
    rubric_df.to_csv(os.path.join(outputs_dir, 'presentation_rubric.csv'), index=False)

    # ---------------- Generate simple PPTX ----------------
    prs = Presentation()

    # Slide 1: Title
    s = prs.slides.add_slide(prs.slide_layouts[0])
    title = s.shapes.title
    subtitle = s.placeholders[1]
    title.text = "Drivee — ML Pricing: Business Case & Presentation Pack"
    subtitle.text = "Автоматически сгенерировано"

    # Slide 2: Executive summary
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Executive summary"
    tx = s.shapes.placeholders[1].text_frame
    tx.clear()
    p = tx.paragraphs[0]
    p.text = f"Текущая ежедневная выручка: {format_rub(defaults['current_daily_revenue'])}"
    p.level = 0
    p2 = tx.add_paragraph()
    p2.text = f"Ожидаемая прибыль от ML (день): {format_rub(daily_ml_profit)} (входные данные)"
    p2.level = 0
    p3 = tx.add_paragraph()
    p3.text = f"Увеличение выручки (пример): {defaults['revenue_uplift_pct']}%"
    p3.level = 0

    # Slide 3: Cost scenarios (table)
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "Cost scenarios & ROI"
    rows = len(summary_df) + 1
    cols = len(summary_df.columns)
    left = Inches(0.5); top = Inches(1.5); width = Inches(9); height = Inches(2.5)
    table = s.shapes.add_table(rows, cols, left, top, width, height).table
    # header
    for j, col in enumerate(summary_df.columns):
        table.cell(0, j).text = col
    # data
    for i in range(len(summary_df)):
        for j, col in enumerate(summary_df.columns):
            table.cell(i+1, j).text = str(summary_df.iloc[i, j])

    # Slide 4: Rubric
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "Оценочная рубрика для презентации"
    rows = len(rubric_df) + 1
    cols = len(rubric_df.columns)
    left = Inches(0.5); top = Inches(1.5); width = Inches(9); height = Inches(2.5)
    table = s.shapes.add_table(rows, cols, left, top, width, height).table
    for j, col in enumerate(rubric_df.columns):
        table.cell(0, j).text = col
    for i in range(len(rubric_df)):
        for j, col in enumerate(rubric_df.columns):
            table.cell(i+1, j).text = str(rubric_df.iloc[i, j])

    # Slide 5: Assets list and next steps
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Assets & Next steps"
    tx = s.shapes.placeholders[1].text_frame
    tx.clear()
    tx.add_paragraph().text = "Сохранённые графики и файлы (см outputs/plots):"
    if price_grid_path and os.path.exists(price_grid_path):
        tx.add_paragraph().text = f" - price grid predictions: {price_grid_path}"
    # add any plots in outputs/plots
    plots = [p for p in os.listdir(os.path.join(outputs_dir, 'plots'))] if os.path.exists(os.path.join(outputs_dir,'plots')) else []
    for p in plots:
        tx.add_paragraph().text = f" - plots/{p}"

    pres_path = os.path.join(outputs_dir, 'presentation.pptx')
    prs.save(pres_path)

    # ---------------- Save some CSV outputs for inspection ----------------
    summary_df.to_csv(os.path.join(outputs_dir, 'summary.csv'), index=False)

    # Also copy price_grid if available to outputs
    if price_grid is not None:
        price_grid.to_csv(os.path.join(outputs_dir, 'price_grid_predictions.csv'), index=False)

    # ---------------- Return paths ----------------
    return {
        'outputs_dir': outputs_dir,
        'summary_csv': os.path.join(outputs_dir, 'summary.csv'),
        'costs_and_roi_csv': os.path.join(outputs_dir, 'costs_and_roi.csv'),
        'rubric_csv': os.path.join(outputs_dir, 'presentation_rubric.csv'),
        'pptx': pres_path
    }


# -------------------------- CLI --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/mnt/data/train.csv', help='Path to train.csv')
    parser.add_argument('--price_grid', type=str, default='/mnt/data/drivee_plots/price_grid_predictions.csv', help='Path to price grid predictions (optional)')
    parser.add_argument('--outdir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    res = generate_presentation(data_path=args.data, price_grid_path=args.price_grid, outputs_dir=args.outdir)
    print('\nGenerated assets:')
    for k,v in res.items():
        print(f" - {k}: {v}")

    print('\nГотово. Откройте outputs/ для файлов презентации и таблиц.')
