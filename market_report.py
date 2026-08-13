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


def generate_ai_summary(market_text, stock_text, risk_text):
    prompt = f"""
你是一名稳健、长期主义风格的全球市场分析师。

请基于以下一周市场数据与波动率数据，写一份简短中文投资市场周报。

要求：
1. 不要给短线交易建议。
2. 不要预测市场一定上涨或下跌。
3. 重点解释市场情绪、风险偏好、科技股表现、美国与非美国市场相对强弱。
4. 结合 VIX 与 VXN 判断当前市场压力属于平静、正常、升温、高压力还是极端压力，并解释两者是否同步。
5. 如果 VXN 明显高于 VIX，可以指出科技股隐含波动率相对更高，但不要把差值机械解释成未来涨跌方向。
6. 可以给长期投资者“维持、正常定投、分批投入预留资金、检查再平衡”等参考。
7. 不得仅凭 VIX/VXN 建议清仓、满仓、一次性重仓或进行短线择时。
8. 语言适合长期指数投资者阅读。
9. 总长度控制在 450-650 中文字。
10. 最后加一句“长期投资者本周可关注：……”。
11. 不要使用 Markdown 表格。

美国与全球主要市场一周表现：
{market_text}

美国头部公司一周表现：
{stock_text}

市场风险温度：
{risk_text}
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

ai_summary = generate_ai_summary(market_text, stock_text, risk_text)

today = datetime.date.today()

plain_body = f"""
投资市场周报
日期：{today}

一、市场风险温度
{risk_text}

二、美国与全球主要市场表现
{market_text}

三、美国头部公司表现
{stock_text}

四、AI 市场解读
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

print("Weekly market report with VIX/VXN risk temperature sent successfully.")
