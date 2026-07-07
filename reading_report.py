import os
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape

from openai import OpenAI


EMAIL_USER = os.environ["EMAIL_USER"]
EMAIL_PASS = os.environ["EMAIL_PASS"]
EMAIL_TO = os.environ.get("READING_EMAIL_TO", os.environ["EMAIL_TO"])
XAI_API_KEY = os.environ["XAI_API_KEY"]

BOOK_REVIEW_FILE = "books_to_review.txt"
BOOK_POOL_FILE = "book_pool.txt"
READING_HISTORY_FILE = "reading_history.txt"

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)


def read_books():
    if not os.path.exists(BOOK_FILE):
        print(f"{BOOK_FILE} not found. No report will be sent.")
        return []

    with open(BOOK_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    books = []
    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        books.append(line)

    return books


def build_books_text(books):
    return "\n".join([f"{i + 1}. {book}" for i, book in enumerate(books)])


def generate_reading_report(books_text):
    prompt = f"""
你是我的私人阅读顾问。你的任务不是替我读书，而是帮我筛选哪些书值得精读、哪些适合泛读、哪些只需要看精华版、哪些暂时跳过。

我的阅读偏好：
- 我喜欢 Nassim Nicholas Taleb 的《黑天鹅》《反脆弱》《随机漫步的傻瓜》。
- 我喜欢 Howard Marks 的《投资最重要的事》和《周期》。
- 我喜欢 MJ DeMarco 的 The Millionaire Fastlane。
- 我喜欢 Robert P. Murphy 的 Lessons for the Young Economist。
- 我喜欢 Malcolm Gladwell 的 Outliers 和 The Tipping Point。
- 我喜欢 Yuval Noah Harari 和 Kevin Kelly 的书。
- 我也喜欢有情感深度和人性复杂度的文学作品，比如 The Kite Runner。

我的核心兴趣：
不确定性、风险、反脆弱、周期、长期投资、财富系统、经济学、技术趋势、社会机制、宏大历史叙事、人性和命运。

请根据下面这批候选书，给我生成一份中文阅读筛选报告。

候选书单：
{books_text}

报告要求：
1. 先给一个总体判断：这批书整体适合我吗？
2. 把每本书分为四类之一：
   A：值得精读
   B：值得泛读
   C：看精华版即可
   D：暂时跳过
3. 每本书都要说明：
   - 推荐等级
   - 一句话判断
   - 为什么适合或不适合我
   - 建议阅读方式
   - 如果没有中文版，建议如何处理
4. 单独列出：
   - 最值得我优先读的 3 本
   - 只看精华即可的书
   - 暂时不建议投入太多时间的书
5. 风格要克制，不要为了鼓励阅读而把每本书都说成必读。
6. 不要编造不存在的信息。如果你不确定某本书是否有中文版，要说“需要进一步确认中文版情况”。
7. 最后给我一个下个月阅读顺序建议。
8. 总长度控制在 1200 到 1800 中文字。
"""

    try:
        response = client.chat.completions.create(
            model="grok-4.3",
            messages=[
                {
                    "role": "system",
                    "content": "你是专业、克制、注重长期价值的私人阅读顾问。"
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=0.35,
            max_tokens=2200,
        )

        print("AI usage:", response.usage)

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Reading report generation failed: {e}")
        return None


def build_html_report(report_text, books):
    books_html = ""
    for book in books:
        books_html += f"""
        <li style="margin-bottom:8px;line-height:1.6;">{escape(book)}</li>
        """

    return f"""
<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:Arial,'Microsoft YaHei',sans-serif;color:#222222;">
    <div style="max-width:780px;margin:0 auto;padding:28px 16px;">
        <div style="background:#1f2937;color:#ffffff;border-radius:18px;padding:28px 24px;margin-bottom:20px;">
            <div style="font-size:13px;color:#d1d5db;margin-bottom:8px;">Personal Reading Report</div>
            <div style="font-size:28px;font-weight:800;line-height:1.3;">个人阅读筛选报告</div>
            <div style="font-size:14px;color:#d1d5db;margin-top:10px;">日期：{datetime.date.today()}</div>
        </div>

        <div style="background:#ffffff;border:1px solid #e8e8e8;border-radius:14px;padding:18px 18px;margin-bottom:22px;">
            <div style="font-size:17px;font-weight:700;margin-bottom:10px;color:#111111;">📚 本期候选书单</div>
            <ol style="padding-left:22px;margin:0;font-size:14px;color:#333333;">
                {books_html}
            </ol>
        </div>

        <div style="background:#ffffff;border:1px solid #e8e8e8;border-radius:14px;padding:20px 20px;margin-bottom:22px;">
            <div style="font-size:17px;font-weight:700;margin-bottom:12px;color:#111111;">🧠 阅读筛选建议</div>
            <div style="font-size:15px;line-height:1.85;color:#333333;white-space:pre-line;">
                {escape(report_text)}
            </div>
        </div>

        <div style="font-size:12px;line-height:1.7;color:#666666;margin-top:24px;padding:14px 16px;background:#ffffff;border:1px solid #e8e8e8;border-radius:12px;">
            <strong>说明：</strong>本邮件为自动生成的阅读筛选报告。它的目的不是替代阅读，而是帮助你决定哪些书值得精读、泛读、看精华或暂时跳过。
        </div>
    </div>
</body>
</html>
"""


def send_email(report_text, books):
    today = datetime.date.today()

    plain_body = f"""
个人阅读筛选报告
日期：{today}

本期候选书单：
{build_books_text(books)}

阅读筛选建议：
{report_text}

说明：
本邮件为自动生成的阅读筛选报告。它的目的不是替代阅读，而是帮助你决定哪些书值得精读、泛读、看精华或暂时跳过。
"""

    html_body = build_html_report(report_text, books)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"个人阅读筛选报告 - {today}"
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO

    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, EMAIL_TO, msg.as_string())

    print("Reading report sent successfully.")


def main():
    books = read_books()

    if not books:
        print("No books found in books_to_review.txt. No AI call and no email sent.")
        return

    books_text = build_books_text(books)
    report_text = generate_reading_report(books_text)

    if report_text is None:
        print("No report generated. Email will not be sent.")
        return

    send_email(report_text, books)


if __name__ == "__main__":
    main()
