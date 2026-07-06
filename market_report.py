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

        # 兼容 yfinance 有时候返回多层表格的情况
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


def generate_ai_summary(market_text, stock_text):
    prompt = f"""
你是一名稳健、长期主义风格的全球市场分析师。

请基于以下一周市场数据，写一份简短中文投资市场周报。

要求：
1. 不要给短线交易建议。
2. 不要预测市场一定上涨或下跌。
3. 重点解释市场情绪、风险偏好、科技股表现、美国与非美国市场相对强弱。
4. 语言适合长期指数投资者阅读。
5. 总长度控制在 350-550 中文字。
6. 最后加一句“长期投资者本周可关注：……”。
7. 不要使用 Markdown 表格。

美国与全球主要市场一周表现：
{market_text}

美国头部公司一周表现：
{stock_text}
"""

    try:
        response = client.chat.completions.create(
            model="grok-4.3",
            messages=[
                {"role": "system", "content": "你是专业但克制的投资市场分析师，表达清晰，不夸张，不给具体买卖建议。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )

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

ai_summary = generate_ai_summary(market_text, stock_text)

today = datetime.date.today()

plain_body = f"""
投资市场周报
日期：{today}

一、美国与全球主要市场表现
{market_text}

二、美国头部公司表现
{stock_text}

三、AI 市场解读
{ai_summary}

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

        {build_table("📊 美国与全球主要市场表现", market_perf)}

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

print("Weekly market report with AI summary sent successfully.")
