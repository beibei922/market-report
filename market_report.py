import os
import datetime
import smtplib
from email.mime.text import MIMEText

import yfinance as yf


EMAIL_USER = os.environ["EMAIL_USER"]
EMAIL_PASS = os.environ["EMAIL_PASS"]
EMAIL_TO = os.environ["EMAIL_TO"]


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

today = datetime.date.today()

simple_summary = """
简短说明：
本周报先采用无 AI 版本，只展示主要市场和美国头部公司的近一周涨跌幅。
后续如果接入 OpenAI API 或其他大模型 API，可以自动生成更自然的中文市场解读。
"""

body = f"""
投资市场周报
日期：{today}

一、美国与全球主要市场表现
{market_text}

二、美国头部公司表现
{stock_text}

三、说明
{simple_summary}

免责声明：
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
