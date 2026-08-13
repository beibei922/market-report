import os
import datetime
import smtplib
import csv
import io
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape

import yfinance as yf
from openai import OpenAI


EMAIL_USER = os.environ["EMAIL_USER"]
EMAIL_PASS = os.environ["EMAIL_PASS"]
EMAIL_TO = os.environ["EMAIL_TO"]
XAI_API_KEY = os.environ["XAI_API_KEY"]

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)


def get_close_series(ticker, period="1y"):
    """下载收盘价序列，并兼容 yfinance 可能返回的多层表格。"""
    data = yf.download(ticker, period=period, progress=False, auto_adjust=True)

    if data.empty:
        return None

    close = data["Close"]

    if hasattr(close, "columns"):
        close = close.iloc[:, 0]

    close = close.dropna()

    if close.empty:
        return None

    return close


def get_weekly_return(ticker):
    try:
        close = get_close_series(ticker, period="7d")

        if close is None or len(close) < 2:
            return None

        start_price = float(close.iloc[0])
        end_price = float(close.iloc[-1])

        return round((end_price / start_price - 1) * 100, 2)

    except Exception as e:
        print(f"Failed to fetch {ticker}: {e}")
        return None


def get_volatility_metrics(ticker):
    """
    返回波动率指数的：
    - 当前值
    - 本周变化
    - 过去一年分位
    """
    try:
        one_year = get_close_series(ticker, period="1y")
        one_week = get_close_series(ticker, period="7d")

        if one_year is None or len(one_year) < 20:
            return {
                "current": None,
                "weekly_change": None,
                "percentile_1y": None,
            }

        current = float(one_year.iloc[-1])

        weekly_change = None
        if one_week is not None and len(one_week) >= 2:
            start_price = float(one_week.iloc[0])
            end_price = float(one_week.iloc[-1])
            weekly_change = round((end_price / start_price - 1) * 100, 2)

        percentile_1y = round(float((one_year <= current).mean() * 100), 1)

        return {
            "current": round(current, 2),
            "weekly_change": weekly_change,
            "percentile_1y": percentile_1y,
        }

    except Exception as e:
        print(f"Failed to fetch volatility metrics for {ticker}: {e}")
        return {
            "current": None,
            "weekly_change": None,
            "percentile_1y": None,
        }


def format_perf_text(perf_dict):
    lines = []
    for name, value in perf_dict.items():
        if value is None:
            lines.append(f"- {name}: 数据不足")
        else:
            sign = "+" if value >= 0 else ""
            lines.append(f"- {name}: {sign}{value}%")
    return "\n".join(lines)


def perf_label(value):
    if value is None:
        return "数据不足"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value}%"


def perf_color(value):
    if value is None:
        return "#666666"
    if value > 0:
        return "#0a7f38"
    if value < 0:
        return "#b42318"
    return "#555555"


def metric_label(value, suffix=""):
    if value is None:
        return "数据不足"
    return f"{value}{suffix}"


def build_table(title, perf_dict):
    rows = ""
    for name, value in perf_dict.items():
        rows += f"""
        <tr>
            <td style="padding:10px 12px;border-bottom:1px solid #eeeeee;color:#222222;">{escape(name)}</td>
            <td style="padding:10px 12px;border-bottom:1px solid #eeeeee;text-align:right;font-weight:700;color:{perf_color(value)};">
                {escape(perf_label(value))}
            </td>
        </tr>
        """

    return f"""
    <div style="margin:22px 0;background:#ffffff;border:1px solid #e8e8e8;border-radius:14px;overflow:hidden;">
        <div style="padding:14px 16px;background:#f7f8fa;border-bottom:1px solid #e8e8e8;font-size:17px;font-weight:700;color:#111111;">
            {escape(title)}
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:14px;">
            {rows}
        </table>
    </div>
    """


def classify_risk(vix, vxn):
    """
    用 VIX/VXN 的过去一年分位数建立一个简单、可解释的市场压力分层。
    这是本周报自定义的观察框架，不是 Cboe 官方评级。
    """
    percentiles = [
        x["percentile_1y"]
        for x in (vix, vxn)
        if x.get("percentile_1y") is not None
    ]

    if not percentiles:
        return {
            "state": "数据不足",
            "emoji": "⚪",
            "score": None,
            "guidance": "本周波动率数据不足，继续按既定长期投资计划执行。",
        }

    score = round(sum(percentiles) / len(percentiles), 1)

    if score >= 90:
        return {
            "state": "极端压力",
            "emoji": "🔴",
            "score": score,
            "guidance": "避免因恐慌而卖出；若本来就有预留资金，可按既定计划分批投入，并检查是否达到再平衡阈值。",
        }
    if score >= 75:
        return {
            "state": "高压力",
            "emoji": "🟠",
            "score": score,
            "guidance": "维持核心仓位和定投；如有预留资金，可考虑分批投入，而不是一次性大幅加仓。",
        }
    if score >= 55:
        return {
            "state": "风险升温",
            "emoji": "🟡",
            "score": score,
            "guidance": "保持既定资产配置和定投节奏，观察风险是否继续扩大，避免因短期波动追涨杀跌。",
        }
    if score >= 20:
        return {
            "state": "正常",
            "emoji": "🟢",
            "score": score,
            "guidance": "维持既定计划，正常定投，不需要仅因波动率做明显仓位调整。",
        }

    return {
        "state": "极低波动",
        "emoji": "🟢",
        "score": score,
        "guidance": "市场非常平静；维持正常定投，不因低波动而额外追高或主动放大风险。",
    }


def build_risk_text(vix, vxn, risk):
    spread = None
    if vix["current"] is not None and vxn["current"] is not None:
        spread = round(vxn["current"] - vix["current"], 2)

    lines = [
        f"- VIX 当前值：{metric_label(vix['current'])}",
        f"- VIX 本周变化：{perf_label(vix['weekly_change'])}",
        f"- VIX 过去一年分位：{metric_label(vix['percentile_1y'], '%')}",
        f"- VXN 当前值：{metric_label(vxn['current'])}",
        f"- VXN 本周变化：{perf_label(vxn['weekly_change'])}",
        f"- VXN 过去一年分位：{metric_label(vxn['percentile_1y'], '%')}",
        f"- VXN - VIX：{metric_label(spread)}",
        f"- 综合状态：{risk['emoji']} {risk['state']}",
        f"- 长期投资者参考：{risk['guidance']}",
    ]
    return "\n".join(lines)


def build_risk_card(vix, vxn, risk):
    spread = None
    if vix["current"] is not None and vxn["current"] is not None:
        spread = round(vxn["current"] - vix["current"], 2)

    def row(label, value):
        return f"""
        <tr>
            <td style="padding:9px 12px;border-bottom:1px solid #eeeeee;color:#444444;">{escape(label)}</td>
            <td style="padding:9px 12px;border-bottom:1px solid #eeeeee;text-align:right;font-weight:700;color:#111111;">
                {escape(value)}
            </td>
        </tr>
        """

    rows = ""
    rows += row("VIX 当前值", metric_label(vix["current"]))
    rows += row("VIX 本周变化", perf_label(vix["weekly_change"]))
    rows += row("VIX 过去一年分位", metric_label(vix["percentile_1y"], "%"))
    rows += row("VXN 当前值", metric_label(vxn["current"]))
    rows += row("VXN 本周变化", perf_label(vxn["weekly_change"]))
    rows += row("VXN 过去一年分位", metric_label(vxn["percentile_1y"], "%"))
    rows += row("VXN - VIX", metric_label(spread))

    score_text = "数据不足" if risk["score"] is None else f"{risk['score']}%"

    return f"""
    <div style="margin:22px 0;background:#ffffff;border:1px solid #e8e8e8;border-radius:14px;overflow:hidden;">
        <div style="padding:14px 16px;background:#f7f8fa;border-bottom:1px solid #e8e8e8;font-size:17px;font-weight:700;color:#111111;">
            🌡️ 市场风险温度
        </div>

        <div style="padding:16px 16px 8px 16px;">
            <div style="font-size:22px;font-weight:800;color:#111111;margin-bottom:6px;">
                {escape(risk['emoji'])} {escape(risk['state'])}
            </div>
            <div style="font-size:13px;color:#666666;">
                VIX / VXN 一年分位综合值：{escape(score_text)}
            </div>
        </div>

        <table style="width:100%;border-collapse:collapse;font-size:14px;">
            {rows}
        </table>

        <div style="padding:14px 16px;background:#fafafa;font-size:14px;line-height:1.7;color:#333333;">
            <strong>长期投资者参考：</strong>{escape(risk['guidance'])}
        </div>

        <div style="padding:10px 16px 14px 16px;background:#fafafa;font-size:11px;line-height:1.6;color:#777777;">
            注：该状态是本周报基于 VIX/VXN 过去一年分位数建立的自定义观察框架，不是官方评级，也不应单独作为买卖依据。
        </div>
    </div>
    """



def get_trend_metrics(ticker):
    """返回价格相对 200 日均线的位置与 200 日均线方向。"""
    try:
        close = get_close_series(ticker, period="18mo")

        if close is None or len(close) < 200:
            return {
                "price": None,
                "ma200": None,
                "distance_pct": None,
                "ma200_slope_20d": None,
                "state": "数据不足",
            }

        ma200 = close.rolling(200).mean().dropna()
        if ma200.empty:
            return {
                "price": None,
                "ma200": None,
                "distance_pct": None,
                "ma200_slope_20d": None,
                "state": "数据不足",
            }

        current_price = float(close.iloc[-1])
        current_ma200 = float(ma200.iloc[-1])
        distance_pct = round((current_price / current_ma200 - 1) * 100, 2)

        ma200_slope_20d = None
        if len(ma200) >= 21:
            old_ma200 = float(ma200.iloc[-21])
            ma200_slope_20d = round((current_ma200 / old_ma200 - 1) * 100, 2)

        if distance_pct > 5:
            state = "强势"
        elif distance_pct >= 0:
            state = "偏强"
        elif distance_pct >= -5:
            state = "偏弱"
        else:
            state = "弱势"

        return {
            "price": round(current_price, 2),
            "ma200": round(current_ma200, 2),
            "distance_pct": distance_pct,
            "ma200_slope_20d": ma200_slope_20d,
            "state": state,
        }

    except Exception as e:
        print(f"Failed to fetch trend metrics for {ticker}: {e}")
        return {
            "price": None,
            "ma200": None,
            "distance_pct": None,
            "ma200_slope_20d": None,
            "state": "数据不足",
        }


def _rate_metrics_from_values(values, source):
    """根据日度 10Y 收益率序列计算当前值、50日均线与20交易日变化。"""
    values = [float(x) for x in values if x is not None]
    if len(values) < 60:
        return None

    current = values[-1]
    ma50 = sum(values[-50:]) / 50
    distance_pct = round((current / ma50 - 1) * 100, 2) if ma50 else None
    change_20d_pp = round(current - values[-21], 2)

    if distance_pct is None:
        state = "数据不足"
    elif distance_pct > 5 or change_20d_pp > 0.35:
        state = "收益率上行"
    elif distance_pct < -5 or change_20d_pp < -0.35:
        state = "收益率下行"
    else:
        state = "收益率相对稳定"

    return {
        "current": round(current, 2),
        "ma50": round(ma50, 2),
        "distance_pct": distance_pct,
        "change_20d_pp": change_20d_pp,
        "state": state,
        "source": source,
    }


def get_rate_metrics(ticker="^TNX"):
    """
    获取美国10年期国债收益率。

    主数据源：FRED DGS10（日度10年期美国国债恒定期限收益率）。
    备用数据源：Yahoo Finance ^TNX。

    这样即使 Yahoo 在 GitHub Actions 中临时取数失败，周报仍可继续获得利率数据。
    """
    fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"

    # 1) 优先 FRED：官方宏观数据源，不需要 API Key。
    try:
        request = urllib.request.Request(
            fred_url,
            headers={"User-Agent": "Mozilla/5.0 market-report/1.0"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            text = response.read().decode("utf-8")

        reader = csv.DictReader(io.StringIO(text))
        values = []
        for row in reader:
            raw = (row.get("DGS10") or "").strip()
            if raw and raw != ".":
                try:
                    values.append(float(raw))
                except ValueError:
                    pass

        result = _rate_metrics_from_values(values, "FRED DGS10")
        if result is not None:
            return result

        print("FRED DGS10 returned insufficient observations; trying Yahoo fallback.")
    except Exception as e:
        print(f"Failed to fetch FRED DGS10: {e}; trying Yahoo fallback.")

    # 2) Yahoo 备用。
    try:
        close = get_close_series(ticker, period="1y")
        if close is not None:
            result = _rate_metrics_from_values(close.tolist(), "Yahoo ^TNX (fallback)")
            if result is not None:
                return result
    except Exception as e:
        print(f"Failed to fetch Yahoo fallback {ticker}: {e}")

    return {
        "current": None,
        "ma50": None,
        "distance_pct": None,
        "change_20d_pp": None,
        "state": "数据不足",
        "source": "FRED / Yahoo 均失败",
    }




def get_valuation_metrics(ticker, neutral_forward_pe):
    """
    获取 ETF 的估值数据。优先使用 Yahoo Finance 的 Forward P/E，
    若不可用则退回 Trailing P/E。

    neutral_forward_pe 是本周报用于比较的简化中性锚点，
    不是官方公允价值，也不是买卖阈值。
    """
    try:
        info = yf.Ticker(ticker).info or {}

        forward_pe = info.get("forwardPE")
        trailing_pe = info.get("trailingPE")

        pe = forward_pe if forward_pe not in (None, 0) else trailing_pe
        pe_type = "Forward P/E" if forward_pe not in (None, 0) else "Trailing P/E"

        if pe in (None, 0):
            return {
                "pe": None,
                "pe_type": "数据不足",
                "neutral_pe": neutral_forward_pe,
                "ratio_to_neutral": None,
                "state": "数据不足",
            }

        pe = float(pe)
        ratio = pe / neutral_forward_pe

        if ratio >= 1.30:
            state = "明显偏贵"
        elif ratio >= 1.15:
            state = "偏贵"
        elif ratio >= 1.00:
            state = "略偏贵"
        elif ratio >= 0.90:
            state = "中性"
        elif ratio >= 0.80:
            state = "偏便宜"
        else:
            state = "明显偏便宜"

        return {
            "pe": round(pe, 2),
            "pe_type": pe_type,
            "neutral_pe": neutral_forward_pe,
            "ratio_to_neutral": round(ratio, 3),
            "state": state,
        }

    except Exception as e:
        print(f"Failed to fetch valuation metrics for {ticker}: {e}")
        return {
            "pe": None,
            "pe_type": "数据不足",
            "neutral_pe": neutral_forward_pe,
            "ratio_to_neutral": None,
            "state": "数据不足",
        }


def score_valuation_metric(valuation):
    """
    估值越便宜，机会分越高。单个指数满分 25。
    这里比较的是当前 P/E 与简化中性锚点的比例。
    """
    ratio = valuation.get("ratio_to_neutral")
    if ratio is None:
        return None
    if ratio >= 1.30:
        return 2
    if ratio >= 1.15:
        return 5
    if ratio >= 1.00:
        return 9
    if ratio >= 0.90:
        return 14
    if ratio >= 0.80:
        return 19
    return 25


def score_trend_distance(distance_pct):
    """趋势越弱，机会温度分越高。满分 35。"""
    if distance_pct is None:
        return None
    if distance_pct >= 15:
        return 2
    if distance_pct >= 10:
        return 5
    if distance_pct >= 5:
        return 8
    if distance_pct >= 0:
        return 12
    if distance_pct >= -5:
        return 18
    if distance_pct >= -10:
        return 25
    if distance_pct >= -20:
        return 31
    return 35


def score_rate_environment(rate):
    """利率越明显上行，股票估值环境越不友好；利率回落则适度提高机会分。满分 25。"""
    if rate.get("distance_pct") is None or rate.get("change_20d_pp") is None:
        return None

    distance = rate["distance_pct"]
    change = rate["change_20d_pp"]

    if distance > 8 or change > 0.50:
        return 3
    if distance > 3 or change > 0.20:
        return 7
    if distance < -8 or change < -0.50:
        return 22
    if distance < -3 or change < -0.20:
        return 18
    return 12


def build_market_dashboard(vix, vxn, risk, spy_trend, qqq_trend, rate, spy_valuation, qqq_valuation):
    """
    V2 市场机会温度：
    - VIX/VXN 情绪压力 30%
    - SPY/QQQ 200 日趋势 25%
    - 估值温度 25%
    - 10 年美债收益率环境 20%

    分数越高 = 市场越偏冷/压力越大/估值越有吸引力，
    并不等于确定的买入信号。
    """
    components = []

    volatility_points = None
    if risk.get("score") is not None:
        volatility_points = round(risk["score"] / 100 * 30, 1)
        components.append((volatility_points, 30))

    trend_scores = [
        score_trend_distance(spy_trend.get("distance_pct")),
        score_trend_distance(qqq_trend.get("distance_pct")),
    ]
    trend_scores = [x for x in trend_scores if x is not None]
    # 原函数的单项满分是35，这里缩放到V2的25分。
    trend_points = round((sum(trend_scores) / len(trend_scores)) / 35 * 25, 1) if trend_scores else None
    if trend_points is not None:
        components.append((trend_points, 25))

    valuation_scores = [
        score_valuation_metric(spy_valuation),
        score_valuation_metric(qqq_valuation),
    ]
    valuation_scores = [x for x in valuation_scores if x is not None]
    valuation_points = round(sum(valuation_scores) / len(valuation_scores), 1) if valuation_scores else None
    if valuation_points is not None:
        components.append((valuation_points, 25))

    old_rate_points = score_rate_environment(rate)
    rate_points = round(old_rate_points / 25 * 20, 1) if old_rate_points is not None else None
    if rate_points is not None:
        components.append((rate_points, 20))

    if not components:
        score = None
    else:
        earned = sum(x[0] for x in components)
        available = sum(x[1] for x in components)
        score = round(earned / available * 100, 1)

    if score is None:
        state = "数据不足"
        emoji = "⚪"
        guidance = "关键数据不足，按既定长期投资计划执行，不基于单周缺失数据调整仓位。"
    elif score < 25:
        state = "偏热 / 平静"
        emoji = "🟢"
        guidance = "维持正常定投，不额外追涨；若有大额待投资现金，可继续分批而非一次性投入。"
    elif score < 50:
        state = "正常"
        emoji = "🔵"
        guidance = "维持既定资产配置与定投节奏，无需因为短期市场波动做明显调整。"
    elif score < 75:
        state = "偏冷 / 压力升高"
        emoji = "🟡"
        guidance = "继续正常定投；若本来就有预留投资现金，可考虑按既定计划适度分批增加投入。"
    else:
        state = "高压力 / 恐慌观察区"
        emoji = "🔴"
        guidance = "避免因恐慌卖出核心长期仓位；如有预留资金，可分批投入并检查资产配置是否需要再平衡。"

    return {
        "score": score,
        "state": state,
        "emoji": emoji,
        "guidance": guidance,
        "volatility_points": volatility_points,
        "trend_points": trend_points,
        "valuation_points": valuation_points,
        "rate_points": rate_points,
    }

def build_dashboard_text(spy_trend, qqq_trend, rate, spy_valuation, qqq_valuation, dashboard):
    score_text = "数据不足" if dashboard["score"] is None else f"{dashboard['score']}/100"
    return "\n".join([
        f"- Market Opportunity Score：{score_text}",
        f"- 综合状态：{dashboard['emoji']} {dashboard['state']}",
        f"- SPY 估值：{metric_label(spy_valuation['pe'])}（{spy_valuation['pe_type']}，{spy_valuation['state']}）",
        f"- QQQ 估值：{metric_label(qqq_valuation['pe'])}（{qqq_valuation['pe_type']}，{qqq_valuation['state']}）",
        f"- SPY vs 200日均线：{metric_label(spy_trend['distance_pct'], '%')}（{spy_trend['state']}）",
        f"- QQQ vs 200日均线：{metric_label(qqq_trend['distance_pct'], '%')}（{qqq_trend['state']}）",
        f"- SPY 200日均线近20日变化：{metric_label(spy_trend['ma200_slope_20d'], '%')}",
        f"- QQQ 200日均线近20日变化：{metric_label(qqq_trend['ma200_slope_20d'], '%')}",
        f"- 美国10年期国债收益率：{metric_label(rate['current'], '%')}（{rate['state']}，来源：{rate.get('source', '未知')}）",
        f"- 10Y vs 50日均线：{metric_label(rate['distance_pct'], '%')}",
        f"- 10Y 近20交易日变化：{metric_label(rate['change_20d_pp'], ' 个百分点')}",
        f"- 长期投资者参考：{dashboard['guidance']}",
    ])

def build_market_dashboard_card(spy_trend, qqq_trend, rate, spy_valuation, qqq_valuation, dashboard):
    def row(label, value):
        return f"""
        <tr>
            <td style="padding:9px 12px;border-bottom:1px solid #eeeeee;color:#444444;">{escape(label)}</td>
            <td style="padding:9px 12px;border-bottom:1px solid #eeeeee;text-align:right;font-weight:700;color:#111111;">
                {escape(value)}
            </td>
        </tr>
        """

    score_text = "数据不足" if dashboard["score"] is None else f"{dashboard['score']}/100"

    rows = ""
    rows += row("SPY 估值", f"{metric_label(spy_valuation['pe'])} · {spy_valuation['pe_type']} · {spy_valuation['state']}")
    rows += row("QQQ 估值", f"{metric_label(qqq_valuation['pe'])} · {qqq_valuation['pe_type']} · {qqq_valuation['state']}")
    rows += row("SPY vs 200日均线", f"{metric_label(spy_trend['distance_pct'], '%')} · {spy_trend['state']}")
    rows += row("QQQ vs 200日均线", f"{metric_label(qqq_trend['distance_pct'], '%')} · {qqq_trend['state']}")
    rows += row("SPY 200日线近20日", metric_label(spy_trend["ma200_slope_20d"], "%"))
    rows += row("QQQ 200日线近20日", metric_label(qqq_trend["ma200_slope_20d"], "%"))
    rows += row("美国10年期国债收益率", f"{metric_label(rate['current'], '%')} · {rate['state']} · {rate.get('source', '未知')}")
    rows += row("10Y vs 50日均线", metric_label(rate["distance_pct"], "%"))
    rows += row("10Y 近20交易日变化", metric_label(rate["change_20d_pp"], " 个百分点"))

    return f"""
    <div style="margin:22px 0;background:#ffffff;border:1px solid #e8e8e8;border-radius:14px;overflow:hidden;">
        <div style="padding:14px 16px;background:#f7f8fa;border-bottom:1px solid #e8e8e8;font-size:17px;font-weight:700;color:#111111;">
            🧭 市场机会仪表盘 · V2.1
        </div>

        <div style="padding:16px 16px 8px 16px;">
            <div style="font-size:22px;font-weight:800;color:#111111;margin-bottom:6px;">
                {escape(dashboard['emoji'])} {escape(dashboard['state'])}
            </div>
            <div style="font-size:14px;color:#555555;margin-bottom:4px;">
                Market Opportunity Score：<strong>{escape(score_text)}</strong>
            </div>
            <div style="font-size:11px;color:#777777;line-height:1.6;">
                V2.1 权重：VIX/VXN 30% · 估值 25% · SPY/QQQ 200日趋势 25% · 10年美债收益率环境 20%
            </div>
        </div>

        <table style="width:100%;border-collapse:collapse;font-size:14px;">
            {rows}
        </table>

        <div style="padding:14px 16px;background:#fafafa;font-size:14px;line-height:1.7;color:#333333;">
            <strong>长期投资者参考：</strong>{escape(dashboard['guidance'])}
        </div>

        <div style="padding:10px 16px 14px 16px;background:#fafafa;font-size:11px;line-height:1.6;color:#777777;">
            注：估值模块优先读取 Yahoo Finance Forward P/E，缺失时退回 Trailing P/E。SPY 20倍、QQQ 25倍仅作为本周报的简化中性估值锚点，不代表公允价值或买卖阈值。该评分不预测短期涨跌。
        </div>
    </div>
    """

def generate_ai_summary(market_text, stock_text, risk_text, dashboard_text):
    prompt = f"""
你是一名稳健、长期主义风格的全球市场分析师。

请基于以下一周市场数据、波动率数据与市场机会仪表盘，写一份简短中文投资市场周报。

要求：
1. 不要给短线交易建议。
2. 不要预测市场一定上涨或下跌。
3. 重点解释市场情绪、风险偏好、科技股表现、美国与非美国市场相对强弱。
4. 结合 VIX/VXN、SPY/QQQ 估值、200日均线趋势和10年美债收益率环境，解释 Market Opportunity Score，而不是只看单一指标。
5. 估值模块使用 ETF 的 Forward P/E（缺失时为 Trailing P/E）；解释估值时保持克制，不把一个 P/E 数字当成绝对买卖信号。
6. 如果 VXN 明显高于 VIX，可以指出科技股隐含波动率相对更高，但不要把差值机械解释成未来涨跌方向。
7. 可以给长期投资者“维持、正常定投、分批投入预留资金、检查再平衡”等参考。
8. 不得仅凭 VIX/VXN 建议清仓、满仓、一次性重仓或进行短线择时。
9. Market Opportunity Score 越高只代表市场越偏冷或压力越大，不等于确定买点；不得把评分解释为短期预测。
10. 语言适合长期指数投资者阅读。
11. 总长度控制在 450-650 中文字。
12. 最后加一句“长期投资者本周可关注：……”。
13. 不要使用 Markdown 表格。

美国与全球主要市场一周表现：
{market_text}

美国头部公司一周表现：
{stock_text}

市场风险温度：
{risk_text}

市场机会仪表盘：
{dashboard_text}
"""

    try:
        response = client.chat.completions.create(
            model="grok-4.3",
            messages=[
                {
                    "role": "system",
                    "content": "你是专业但克制的投资市场分析师，表达清晰，不夸张，不把波动率指标当成机械买卖信号。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )

        print(f"AI usage: {response.usage}")
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"AI summary failed: {e}")
        return "AI 市场解读生成失败。本期先展示市场数据与风险温度。请检查 xAI API Key、模型权限或账户余额。"


markets = {
    "🇺🇸 S&P 500": "^GSPC",
    "🇺🇸 Nasdaq 100": "^NDX",
    "🇺🇸 Dow Jones": "^DJI",
    "🇺🇸 Russell 2000": "^RUT",
    "🇪🇺 Europe STOXX 50": "^STOXX50E",
    "🇩🇪 Germany DAX": "^GDAXI",
    "🇯🇵 Japan Nikkei 225": "^N225",
    "🇭🇰 Hong Kong Hang Seng": "^HSI",
    "🇮🇳 India Nifty 50": "^NSEI",
}

stocks = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Nvidia": "NVDA",
    "Amazon": "AMZN",
    "Alphabet / Google": "GOOGL",
    "Meta": "META",
    "Tesla": "TSLA",
    "Berkshire Hathaway": "BRK-B",
    "JPMorgan Chase": "JPM",
    "Eli Lilly": "LLY",
}


market_perf = {name: get_weekly_return(ticker) for name, ticker in markets.items()}
stock_perf = {name: get_weekly_return(ticker) for name, ticker in stocks.items()}

market_text = format_perf_text(market_perf)
stock_text = format_perf_text(stock_perf)

vix = get_volatility_metrics("^VIX")
vxn = get_volatility_metrics("^VXN")
risk = classify_risk(vix, vxn)
risk_text = build_risk_text(vix, vxn, risk)

spy_trend = get_trend_metrics("SPY")
qqq_trend = get_trend_metrics("QQQ")
rate = get_rate_metrics("^TNX")

# V2 估值模块：优先使用 Forward P/E。
# 20（SPY）与 25（QQQ）是本周报的简化中性锚点，仅用于温度比较。
spy_valuation = get_valuation_metrics("SPY", neutral_forward_pe=20.0)
qqq_valuation = get_valuation_metrics("QQQ", neutral_forward_pe=25.0)

dashboard = build_market_dashboard(
    vix, vxn, risk, spy_trend, qqq_trend, rate, spy_valuation, qqq_valuation
)
dashboard_text = build_dashboard_text(
    spy_trend, qqq_trend, rate, spy_valuation, qqq_valuation, dashboard
)

ai_summary = generate_ai_summary(market_text, stock_text, risk_text, dashboard_text)

today = datetime.date.today()

plain_body = f"""
投资市场周报
日期：{today}

一、市场风险温度
{risk_text}

二、市场机会仪表盘
{dashboard_text}

三、美国与全球主要市场表现
{market_text}

四、美国头部公司表现
{stock_text}

五、AI 市场解读
{ai_summary}

免责声明：
本邮件为自动生成的市场信息整理，不构成投资建议。
VIX/VXN 风险状态为本周报自定义观察框架，不是官方评级，也不应单独作为买卖依据。
"""

html_body = f"""
<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:Arial,'Microsoft YaHei',sans-serif;color:#222222;">
    <div style="max-width:760px;margin:0 auto;padding:28px 16px;">
        <div style="background:#111827;color:#ffffff;border-radius:18px;padding:28px 24px;margin-bottom:20px;">
            <div style="font-size:13px;color:#d1d5db;margin-bottom:8px;">Weekly Market Report</div>
            <div style="font-size:28px;font-weight:800;line-height:1.3;">投资市场周报</div>
            <div style="font-size:14px;color:#d1d5db;margin-top:10px;">日期：{today}</div>
        </div>

        {build_risk_card(vix, vxn, risk)}

        {build_market_dashboard_card(spy_trend, qqq_trend, rate, spy_valuation, qqq_valuation, dashboard)}

        <div style="background:#ffffff;border:1px solid #e8e8e8;border-radius:14px;padding:18px 18px;margin-bottom:22px;">
            <div style="font-size:17px;font-weight:700;margin-bottom:10px;color:#111111;">🧠 AI 市场解读</div>
            <div style="font-size:15px;line-height:1.8;color:#333333;white-space:pre-line;">
                {escape(ai_summary)}
            </div>
        </div>

        {build_table("📊 美国与全球主要市场表现", market_perf)}

        {build_table("🏢 美国头部公司表现", stock_perf)}

        <div style="font-size:12px;line-height:1.7;color:#666666;margin-top:24px;padding:14px 16px;background:#ffffff;border:1px solid #e8e8e8;border-radius:12px;">
            <strong>免责声明：</strong>本邮件为自动生成的市场信息整理，不构成投资建议。数据来自公开市场数据源，可能存在延迟或缺失。VIX/VXN 风险状态为本周报自定义观察框架，不是官方评级，也不应单独作为买卖依据。
        </div>
    </div>
</body>
</html>
"""

msg = MIMEMultipart("alternative")
msg["Subject"] = f"投资市场周报 - {today}"
msg["From"] = EMAIL_USER
msg["To"] = EMAIL_TO

msg.attach(MIMEText(plain_body, "plain", "utf-8"))
msg.attach(MIMEText(html_body, "html", "utf-8"))

with smtplib.SMTP("smtp.gmail.com", 587) as server:
    server.starttls()
    server.login(EMAIL_USER, EMAIL_PASS)
    server.sendmail(EMAIL_USER, EMAIL_TO, msg.as_string())

print("Weekly market report with VIX/VXN + valuation + market opportunity dashboard V2.1 sent successfully.")
