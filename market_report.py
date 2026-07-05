import os
import datetime
import smtplib
from email.mime.text import MIMEText

import yfinance as yf
from openai import OpenAI


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMAIL_USER = os.environ["EMAIL_USER"]
EMAIL_PASS = os.environ["EMAIL_PASS"]
EMAIL_TO = os.environ["EMAIL_TO"]

client = OpenAI(api_key=OPENAI_API_KEY)


def get_weekly_return(ticker):
    data = yf.download(ticker, period="7d", progress=False)

    if data.empty or len(data) < 2:
        return None

    start_price = float(data["Close"].iloc[0])
    end_price = float(data["Close"].iloc[-1])

    return round((end_price / start_price - 1) * 100, 2)


def format_perf(perf_dict):
    lines = []
    for name, value in perf_dict.items():
        if value is None:
            lines.append(f"- {name}: 数据不足")
        else:
            sign = "+" if value >= 0 else ""
            lines.append(f"- {name}: {sign}{value}%")
    return "\n".join(lines)


markets = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "Europe STOXX 50": "^STOXX50E",
    "Germany DAX": "^GDAXI",
    "Japan Nikkei 225": "^N225",
    "Hong Kong Hang Seng": "^HSI",
    "India Nifty 50": "^NSEI",
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

market_text = format_perf(market_perf)
stock_text = format_perf(stock_perf)

prompt = f"""
你是一名稳健、长期主义风格的全球市场分析师。

请基于以下一周市场数据，写一份简短中文投资市场周报。

要求：
1. 不要给短线交易建议。
2. 不要预测市场一定上涨或下跌。
3. 重点解释市场情绪、风险偏好、科技股表现、全球主要市场相对强弱。
4. 语言适合长期指数投资者阅读。
5. 总长度控制在 400-600 中文字。

美国与全球主要市场一周表现：
{market_text}

美国头部公司一周表现：
{stock_text}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "你是专业但克制的投资市场分析师。"},
        {"role": "user", "content": prompt},
    ],
)

analysis = response.choices[0].message.content

today = datetime.date.today()

body = f"""
投资市场周报
日期：{today}

一、美国与全球主要市场表现
{market_text}

二、美国头部公司表现
{stock_text}

三、简短市场解读
{analysis}

说明：
本邮件为自动生成的市场信息整理，不构成投资建议。
"""

msg = MIMEText(body, "plain", "utf-8")
msg["Subject"] = f"投资市场周报 - {today}"
msg["From"] = EMAIL_USER
msg["To"] = EMAIL_TO

with smtplib.SMTP("smtp.gmail.com", 587) as server:
    server.starttls()
    server.login(EMAIL_USER, EMAIL_PASS)
    server.sendmail(EMAIL_USER, EMAIL_TO, msg.as_string())

print("Weekly market report sent successfully.")
