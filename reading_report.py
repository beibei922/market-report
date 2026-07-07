import os
import datetime
import smtplib
import hashlib
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


def normalize_book_name(book):
    return " ".join(book.lower().replace("—", "-").split())


def read_lines_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"{file_path} not found.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clean_lines = []
    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        clean_lines.append(line)

    return clean_lines


def read_books_to_review():
    return read_lines_from_file(BOOK_REVIEW_FILE)


def read_book_pool():
    return read_lines_from_file(BOOK_POOL_FILE)


def read_history_books():
    lines = read_lines_from_file(READING_HISTORY_FILE)
    history_books = set()

    for line in lines:
        parts = [p.strip() for p in line.split("|")]

        if len(parts) >= 3:
            book = parts[-1]
        else:
            book = line

        history_books.add(normalize_book_name(book))

    return history_books


def remove_duplicates(books):
    seen = set()
    result = []

    for book in books:
        key = normalize_book_name(book)

        if key in seen:
            continue

        seen.add(key)
        result.append(book)

    return result


def filter_history_books(book_pool, history_books):
    result = []

    for book in book_pool:
        key = normalize_book_name(book)

        if key in history_books:
            continue

        result.append(book)

    return result


def select_monthly_candidates(available_books, max_candidates=70):
    """
    为了节省 tokens，不把整个 book_pool 都发给模型。
    每个月从候选池里轮换选一批书，让模型在这批里面推荐。
    """

    if len(available_books) <= max_candidates:
        return available_books

    today = datetime.date.today()
    month_key = today.year * 12 + today.month

    # 用月份做一个稳定的轮换起点
    start_index = month_key % len(available_books)

    rotated = available_books[start_index:] + available_books[:start_index]

    return rotated[:max_candidates]


def build_books_text(books):
    return "\n".join([f"{i + 1}. {book}" for i, book in enumerate(books)])


def generate_manual_review_report(books_text):
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

    return call_grok(prompt, max_tokens=2200)


def generate_auto_recommendation_report(candidate_books_text):
    prompt = f"""
你是我的私人阅读顾问。你的任务不是替我读书，而是帮我从候选书库中挑选本月最值得关注的书。

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

请从下面这批候选书中，给我生成一份“本月阅读推荐报告”。

候选书：
{candidate_books_text}

报告要求：
1. 推荐总数控制在 6 到 8 本。
2. 必须分成四类：
   A：最值得精读，推荐 2 本
   B：值得泛读，推荐 2 到 3 本
   C：看精华版即可，推荐 1 到 2 本
   D：暂时不建议投入太多时间，列出 1 到 2 本
3. 每本书说明：
   - 推荐等级
   - 一句话判断
   - 为什么适合或不适合我
   - 建议阅读方式
   - 是否可能有中文版；如果不确定，请说“需要进一步确认中文版情况”
4. 最后单独给出：
   - 如果本月只能读一本，我建议读哪本
   - 本月阅读顺序
   - 哪本适合以后再读
5. 风格克制，不要把每本书都说成必读。
6. 不要编造不存在的信息。
7. 总长度控制在 1000 到 1500 中文字。
"""

    return call_grok(prompt, max_tokens=1900)


def call_grok(prompt, max_tokens):
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
            max_tokens=max_tokens,
        )

        print("AI usage:", response.usage)

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Reading report generation failed: {e}")
        return None


def build_html_report(report_text, displayed_books, report_mode, extra_note):
    books_html = ""
    for book in displayed_books:
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
            <div style="font-size:17px;font-weight:700;margin-bottom:10px;color:#111111;">📌 报告模式</div>
            <div style="font-size:14px;line-height:1.8;color:#333333;">
                {escape(report_mode)}
                <br>
                {escape(extra_note)}
            </div>
        </div>

        <div style="background:#ffffff;border:1px solid #e8e8e8;border-radius:14px;padding:18px 18px;margin-bottom:22px;">
            <div style="font-size:17px;font-weight:700;margin-bottom:10px;color:#111111;">📚 本期参考书单</div>
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


def send_email(report_text, displayed_books, report_mode, extra_note):
    today = datetime.date.today()

    plain_body = f"""
个人阅读筛选报告
日期：{today}

报告模式：
{report_mode}

说明：
{extra_note}

本期参考书单：
{build_books_text(displayed_books)}

阅读筛选建议：
{report_text}

说明：
本邮件为自动生成的阅读筛选报告。它的目的不是替代阅读，而是帮助你决定哪些书值得精读、泛读、看精华或暂时跳过。
"""

    html_body = build_html_report(report_text, displayed_books, report_mode, extra_note)

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
    manual_books = read_books_to_review()

    if manual_books:
        manual_books = remove_duplicates(manual_books)
        books_text = build_books_text(manual_books)

        report_text = generate_manual_review_report(books_text)

        if report_text is None:
            print("No report generated. Email will not be sent.")
            return

        send_email(
            report_text=report_text,
            displayed_books=manual_books,
            report_mode="手动候选书单评估模式",
            extra_note="系统检测到 books_to_review.txt 中有候选书，因此优先评估你手动添加的书单。"
        )

        return

    book_pool = read_book_pool()

    if not book_pool:
        print("No manual books and no book_pool.txt content. No AI call and no email sent.")
        return

    book_pool = remove_duplicates(book_pool)
    history_books = read_history_books()
    available_books = filter_history_books(book_pool, history_books)

    if not available_books:
        print("All books in book_pool.txt appear to be in reading_history.txt. No email sent.")
        return

    monthly_candidates = select_monthly_candidates(available_books, max_candidates=70)
    candidate_books_text = build_books_text(monthly_candidates)

    report_text = generate_auto_recommendation_report(candidate_books_text)

    if report_text is None:
        print("No report generated. Email will not be sent.")
        return

    send_email(
        report_text=report_text,
        displayed_books=monthly_candidates,
        report_mode="自动候选书库推荐模式",
        extra_note="系统检测到 books_to_review.txt 为空，因此从 book_pool.txt 中筛选候选书，并排除了 reading_history.txt 中已记录的书。"
    )


if __name__ == "__main__":
    main()
