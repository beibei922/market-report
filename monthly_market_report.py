import os
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape

import yfinance as yf
from openai import OpenAI

EMAIL_USER = os.environ["EMAIL_USER"]
EMAIL_PASS = os.environ["EMAIL_PASS"]
EMAIL_TO = os.environ["EMAIL_TO"]
XAI_API_KEY = os.environ["XAI_API_KEY"]

client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")


def get_close_series(ticker, period="1y"):
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data.empty:
            return None
        close = data["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        close = close.dropna()
        return close if not close.empty else None
    except Exception as e:
        print(f"Failed to fetch {ticker}: {e}")
        return None


def get_monthly_return(ticker):
    try:
        close = get_close_series(ticker, period="1mo")
        if close is None or len(close) < 2:
            return None
        return round((float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100, 2)
    except Exception as e:
        print(f"Failed monthly return for {ticker}: {e}")
        return None


def get_volatility_metrics(ticker):
    try:
        one_year = get_close_series(ticker, period="1y")
        one_month = get_close_series(ticker, period="1mo")
        if one_year is None or len(one_year) < 20:
            return {"current": None, "monthly_change": None, "percentile_1y": None}

        current = float(one_year.iloc[-1])
        monthly_change = None
        if one_month is not None and len(one_month) >= 2:
            monthly_change = round((float(one_month.iloc[-1]) / float(one_month.iloc[0]) - 1) * 100, 2)

        percentile = round(float((one_year <= current).mean() * 100), 1)
        return {
            "current": round(current, 2),
            "monthly_change": monthly_change,
            "percentile_1y": percentile,
        }
    except Exception as e:
        print(f"Failed volatility metrics for {ticker}: {e}")
        return {"current": None, "monthly_change": None, "percentile_1y": None}


def perf_label(value):
    if value is None:
        return "数据不足"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value}%"


def metric_label(value, suffix=""):
    return "数据不足" if value is None else f"{value}{suffix}"


def perf_color(value):
    if value is None:
        return "#666666"
    return "#0a7f38" if value > 0 else "#b42318" if value < 0 else "#555555"


def format_perf_text(perf_dict):
    return "\n".join(f"- {name}: {perf_label(value)}" for name, value in perf_dict.items())


def build_table(title, perf_dict):
    rows = ""
    for name, value in perf_dict.items():
        rows += f'''<tr>
<td style="padding:10px 12px;border-bottom:1px solid #eeeeee;color:#222222;">{escape(name)}</td>
<td style="padding:10px 12px;border-bottom:1px solid #eeeeee;text-align:right;font-weight:700;color:{perf_color(value)};">{escape(perf_label(value))}</td>
</tr>'''
    return f'''<div style="margin:22px 0;background:#ffffff;border:1px solid #e8e8e8;border-radius:14px;overflow:hidden;">
<div style="padding:14px 16px;background:#f7f8fa;border-bottom:1px solid #e8e8e8;font-size:17px;font-weight:700;color:#111111;">{escape(title)}</div>
<table style="width:100%;border-collapse:collapse;font-size:14px;">{rows}</table>
</div>'''


def classify_risk(vix, vxn):
    vals = [x["percentile_1y"] for x in (vix, vxn) if x.get("percentile_1y") is not None]
    if not vals:
        return "⚪", "数据不足", None
    score = round(sum(vals) / len(vals), 1)
    if score >= 90:
        return "🔴", "极端压力", score
    if score >= 75:
        return "🟠", "高压力", score
    if score >= 55:
        return "🟡", "风险升温", score
    if score >= 20:
        return "🟢", "正常", score
    return "🟢", "极低波动", score


def build_risk_text(vix, vxn):
    emoji, state, score = classify_risk(vix, vxn)
    spread = None
    if vix["current"] is not None and vxn["current"] is not None:
        spread = round(vxn["current"] - vix["current"], 2)
    score_text = "数据不足" if score is None else f"{score}%"
    return f'''VIX 当前值：{metric_label(vix['current'])}
VIX 月度变化：{perf_label(vix['monthly_change'])}
VIX 过去一年分位：{metric_label(vix['percentile_1y'], '%')}
VXN 当前值：{metric_label(vxn['current'])}
VXN 月度变化：{perf_label(vxn['monthly_change'])}
VXN 过去一年分位：{metric_label(vxn['percentile_1y'], '%')}
VXN - VIX：{metric_label(spread)}
综合状态：{emoji} {state}
综合分位：{score_text}'''


def build_structure_text(market_perf):
    sp = market_perf.get("🇺🇸 S&P 500")
    ndx = market_perf.get("🇺🇸 Nasdaq 100")
    rut = market_perf.get("🇺🇸 Russell 2000")
    non_us = [market_perf.get(k) for k in ["🇪🇺 Europe STOXX 50", "🇯🇵 Japan Nikkei 225", "🇭🇰 Hong Kong Hang Seng"]]
    non_us = [x for x in non_us if x is not None]
    non_us_avg = round(sum(non_us) / len(non_us), 2) if non_us else None

    tech_vs_sp = round(ndx - sp, 2) if ndx is not None and sp is not None else None
    small_vs_sp = round(rut - sp, 2) if rut is not None and sp is not None else None
    us_vs_non_us = round(sp - non_us_avg, 2) if sp is not None and non_us_avg is not None else None

    return f'''科技 vs 大盘（Nasdaq 100 - S&P 500）：{perf_label(tech_vs_sp)}
小盘 vs 大盘（Russell 2000 - S&P 500）：{perf_label(small_vs_sp)}
美国 vs 非美（S&P 500 - 欧洲/日本/香港平均）：{perf_label(us_vs_non_us)}'''


def generate_ai_report(market_text, risk_text, structure_text):
    prompt = f'''你是一名稳健、长期主义风格的全球市场分析师。

请根据下面的数据，写一份简洁的中文“月度投资报告”。

严格按以下结构输出：

【本月一句话总结】
1-2句，不超过80字。

【趋势与结构】
最多3条，覆盖：大盘 vs 科技、大盘股 vs 小盘股、美国 vs 非美。

【本月3个关键变化】
严格最多3条，只写真正重要的变化，不堆砌新闻。

【长期投资者结论】
只写4行：
核心仓位：保持 / 检查再平衡
定投：正常 / 稍放缓 / 可适度加快
额外现金：等待 / 分批投入
下月重点观察：一句话

规则：
- 不给短线交易建议。
- 不预测市场一定上涨或下跌。
- 不因 VIX/VXN 单独建议清仓、满仓或一次性重仓。
- VIX/VXN 仅用于判断市场压力与风险情绪，不作为机械买卖信号。
- 总长度控制在 500-750 中文字。
- 不使用 Markdown 表格。

主要市场月度表现：
{market_text}

市场风险温度：
{risk_text}

趋势与结构数据：
{structure_text}
'''
    try:
        response = client.chat.completions.create(
            model="grok-4.3",
            messages=[
                {"role": "system", "content": "你是专业、克制、长期主义的投资市场分析师。重视风险管理与资产配置，不做短线预测。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
        )
        print(f"AI usage: {response.usage}")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"AI monthly report failed: {e}")
        return "AI 月度解读生成失败。本期先展示市场数据。"


markets = {
    "🇺🇸 S&P 500": "^GSPC",
    "🇺🇸 Nasdaq 100": "^NDX",
    "🇺🇸 Russell 2000": "^RUT",
    "🇪🇺 Europe STOXX 50": "^STOXX50E",
    "🇯🇵 Japan Nikkei 225": "^N225",
    "🇭🇰 Hong Kong Hang Seng": "^HSI",
}

market_perf = {name: get_monthly_return(ticker) for name, ticker in markets.items()}
market_text = format_perf_text(market_perf)

vix = get_volatility_metrics("^VIX")
vxn = get_volatility_metrics("^VXN")
risk_text = build_risk_text(vix, vxn)
structure_text = build_structure_text(market_perf)
ai_report = generate_ai_report(market_text, risk_text, structure_text)

today = datetime.date.today()
report_month = (today.replace(day=1) - datetime.timedelta(days=1)).strftime("%Y-%m")

plain_body = f'''月度投资报告
报告月份：{report_month}

一、本月市场概览
{market_text}

二、市场风险温度
{risk_text}

三、趋势与结构
{structure_text}

四、本月关键变化与长期投资者结论
{ai_report}

免责声明：
本邮件为自动生成的市场信息整理，不构成投资建议。
'''

emoji, risk_state, risk_score = classify_risk(vix, vxn)
risk_score_text = "数据不足" if risk_score is None else f"{risk_score}%"

html_body = f'''<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:Arial,'Microsoft YaHei',sans-serif;color:#222222;">
<div style="max-width:760px;margin:0 auto;padding:28px 16px;">
<div style="background:#111827;color:#ffffff;border-radius:18px;padding:28px 24px;margin-bottom:20px;">
<div style="font-size:13px;color:#d1d5db;margin-bottom:8px;">Monthly Investment Report</div>
<div style="font-size:28px;font-weight:800;line-height:1.3;">月度投资报告</div>
<div style="font-size:14px;color:#d1d5db;margin-top:10px;">报告月份：{report_month}</div>
</div>

{build_table("📊 本月市场概览", market_perf)}

<div style="background:#ffffff;border:1px solid #e8e8e8;border-radius:14px;padding:18px;margin-bottom:22px;">
<div style="font-size:17px;font-weight:700;margin-bottom:10px;">🌡️ 市场风险温度</div>
<div style="font-size:22px;font-weight:800;margin-bottom:10px;">{escape(emoji)} {escape(risk_state)}</div>
<div style="font-size:13px;color:#666666;margin-bottom:12px;">VIX / VXN 一年分位综合值：{escape(risk_score_text)}</div>
<div style="font-size:14px;line-height:1.8;white-space:pre-line;">{escape(risk_text)}</div>
</div>

<div style="background:#ffffff;border:1px solid #e8e8e8;border-radius:14px;padding:18px;margin-bottom:22px;">
<div style="font-size:17px;font-weight:700;margin-bottom:10px;">🔍 趋势与结构</div>
<div style="font-size:14px;line-height:1.8;white-space:pre-line;">{escape(structure_text)}</div>
</div>

<div style="background:#ffffff;border:1px solid #e8e8e8;border-radius:14px;padding:18px;margin-bottom:22px;">
<div style="font-size:17px;font-weight:700;margin-bottom:10px;">🧠 本月关键变化 & 长期投资者结论</div>
<div style="font-size:15px;line-height:1.85;color:#333333;white-space:pre-line;">{escape(ai_report)}</div>
</div>

<div style="font-size:12px;line-height:1.7;color:#666666;margin-top:24px;padding:14px 16px;background:#ffffff;border:1px solid #e8e8e8;border-radius:12px;">
<strong>免责声明：</strong>本邮件为自动生成的市场信息整理，不构成投资建议。数据来自公开市场数据源，可能存在延迟或缺失。
</div>
</div>
</body>
</html>'''

msg = MIMEMultipart("alternative")
msg["Subject"] = f"月度投资报告 - {report_month}"
msg["From"] = EMAIL_USER
msg["To"] = EMAIL_TO
msg.attach(MIMEText(plain_body, "plain", "utf-8"))
msg.attach(MIMEText(html_body, "html", "utf-8"))

with smtplib.SMTP("smtp.gmail.com", 587) as server:
    server.starttls()
    server.login(EMAIL_USER, EMAIL_PASS)
    server.sendmail(EMAIL_USER, EMAIL_TO, msg.as_string())

print("Monthly investment report sent successfully.")
