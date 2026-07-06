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


def get_weekly_return(ticker):
    try:
        data = yf.download(ticker, period="7d", progress=False, auto_adjust=True)

        if data.empty or len(data) < 2:
            return None

        close = data["Close"]

        if hasattr(close, "columns"):
            close = close.iloc[:, 0]

        close = close.dropna()

        if len(close) < 2:
            return None

        start_price = float(close.iloc[0])
        end_price = float(close.iloc[-1])

        return round((end_price / start_price - 1) * 100, 2)

    except Exception as e:
        print(f"Failed to fetch {ticker}: {e}")
        return None


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


def format_perf_text(perf_dict):
    lines = []
    for name, value in perf_dict.items():
        lines.append(f"{name}: {perf_label(value)}")
    return "\n".join(lines)


def get_movers(perf_dict, top_n=2):
    valid_items = [(name, value) for name, value in perf_dict.items() if value is not None]

    if not valid_items:
        return "数据不足"

    sorted_items = sorted(valid_items, key=lambda x: x[1])
    losers = sorted_items[:top_n]
    gainers = sorted_items[-top_n:][::-1]

    gainers_text = "；".join([f"{name} {perf_label(value)}" for name, value in gainers])
    losers_text = "；".join([f"{name} {perf_label(value)}" for name, value in losers])

    return f"涨幅较大：{gainers_text}\n跌幅较大：{losers_text}"


def build_table(title, perf_dict):
    rows = ""

    for name, value in perf_dict.items():
        rows += f"""
        <tr>
            <td style="padding:10px 12px;border-bottom:1px solid #eeeeee;color:#222222;">
                {escape(name)}
            </td>
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


def generate_ai_summary(market_brief, sector_movers, stock_movers):
    prompt = f"""
请用中文写一段简短投资市场周报，面向长期指数投资者。

要求：
- 250到400字。
- 不给买卖建议。
- 不预测市场一定涨跌。
- 点评美国大盘、非美国主要市场、头部公司表现。
- 必须提到本周异常波动：哪些板块或个股涨跌较明显。
- 对异常波动只做克制解释，不编造具体新闻原因。
- 最后加一句：长期投资者本周可关注：……

主要市场：
{market_brief}

板块异常：
{sector_movers}

头部公司异常：
{stock_movers}
"""

    try:
        response = client.chat.completions.create(
            model="grok-4.3",
            messages=[
                {
                    "role": "system",
                    "content": "你是专业但克制的市场分析师，语言简洁，不夸张。"
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=0.3,
            max_tokens=500,
        )

        print("AI usage:", response.usage)

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"AI summary failed: {e}")
        return "AI 市场解读生成失败。本期先展示市场数据。请检查 xAI API Key、模型权限或账户余额。"


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

sectors = {
    "Technology / 科技": "XLK",
    "Financials / 金融": "XLF",
    "Health Care / 医疗": "XLV",
    "Energy / 能源": "XLE",
    "Consumer Discretionary / 可选消费": "XLY",
    "Consumer Staples / 必需消费": "XLP",
    "Industrials / 工业": "XLI",
    "Communication Services / 通讯服务": "XLC",
    "Utilities / 公用事业": "XLU",
    "Real Estate / 房地产": "XLRE",
    "Materials / 材料": "XLB",
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
sector_perf = {name: get_weekly_return(ticker) for name, ticker in sectors.items()}
stock_perf = {name: get_weekly_return(ticker) for name, ticker in stocks.items()}

market_text = format_perf_text(market_perf)
sector_text = format_perf_text(sector_perf)
stock_text = format_perf_text(stock_perf)

sector_movers = get_movers(sector_perf, top_n=2)
stock_movers = get_movers(stock_perf, top_n=2)

ai_summary = generate_ai_summary(
    market_brief=market_text,
    sector_movers=sector_movers,
    stock_movers=stock_movers,
)

today = datetime.date.today()

plain_body = f"""
投资市场周报
日期：{today}

一、AI 市场解读
{ai_summary}

二、美国与全球主要市场表现
{market_text}

三、美国主要板块表现
{sector_text}

四、美国头部公司表现
{stock_text}

免责声明：
本邮件为自动生成的市场信息整理，不构成投资建议。
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

        <div style="background:#ffffff;border:1px solid #e8e8e8;border-radius:14px;padding:18px 18px;margin-bottom:22px;">
            <div style="font-size:17px;font-weight:700;margin-bottom:10px;color:#111111;">🧠 AI 市场解读</div>
            <div style="font-size:15px;line-height:1.8;color:#333333;white-space:pre-line;">
                {escape(ai_summary)}
            </div>
        </div>

        <div style="background:#ffffff;border:1px solid #e8e8e8;border-radius:14px;padding:18px 18px;margin-bottom:22px;">
            <div style="font-size:17px;font-weight:700;margin-bottom:10px;color:#111111;">⚠️ 本周异常波动</div>
            <div style="font-size:14px;line-height:1.8;color:#333333;white-space:pre-line;">
板块：
{escape(sector_movers)}

头部公司：
{escape(stock_movers)}
            </div>
        </div>

        {build_table("📊 美国与全球主要市场表现", market_perf)}

        {build_table("🏭 美国主要板块表现", sector_perf)}

        {build_table("🏢 美国头部公司表现", stock_perf)}

        <div style="font-size:12px;line-height:1.7;color:#666666;margin-top:24px;padding:14px 16px;background:#ffffff;border:1px solid #e8e8e8;border-radius:12px;">
            <strong>免责声明：</strong>本邮件为自动生成的市场信息整理，不构成投资建议。数据来自公开市场数据源，可能存在延迟或缺失。
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

print("Weekly market report with sector movers sent successfully.")
