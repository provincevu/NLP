from openai import OpenAI

stream = True

client = OpenAI(
    base_url="http://127.0.0.1:5001/v1",
    api_key="sk-6af49da8d0b54065a2f7929ac7a43512"
)

system_prompt = """
Bạn là AI chuyên tạo synthetic query cho hệ thống retrieval/RAG.

Nhiệm vụ:
- Dựa chủ yếu vào passage để sinh thêm nhiều câu hỏi mới phù hợp.
- Các câu hỏi phải có thể được trả lời trực tiếp từ passage.
- Không thêm thông tin ngoài passage.
- Không thay đổi ý nghĩa của passage.

Danh sách câu hỏi hiện có chỉ dùng để tham khảo nhằm:
- tránh lặp lại
- tránh tạo câu hỏi quá giống nhau
- tăng độ đa dạng cách hỏi

Ưu tiên:
- đa dạng cách diễn đạt
- nhiều kiểu diễn đạt tự nhiên
- câu hỏi ngắn
- câu hỏi đời thường
- câu hỏi kiểu sinh viên
- câu hỏi dùng từ đồng nghĩa
- câu hỏi viết tắt tự nhiên nếu phù hợp

Yêu cầu:
- Không sinh câu hỏi trùng hoặc gần giống với câu hỏi đã có.
- Mỗi câu hỏi phải khác biệt rõ ràng về wording.
- Chỉ trả về danh sách câu hỏi.
- Không đánh số.
- Mỗi dòng là một câu hỏi.
- Không giải thích.
"""

user_prompt = """
Passage:
"Điểm thành phần A phải được nhập vào hệ thống EduSoft.Net đúng thời gian quy định."

Existing questions:
- điểm thành phần A nhập vào EduSoft.Net khi nào
- khi nào cần nhập điểm thành phần A
- thời hạn nhập điểm A là gì

Generate as many diverse and meaningful questions as the passage can naturally support.

Only generate questions that are directly and clearly answerable from the passage.

Try to maximize diversity in wording and phrasing without changing the original intent.

Do not invent new information, infer unsupported facts, or force additional questions once meaningful variations are exhausted.

Quality and semantic correctness are more important than quantity.
"""

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ],
    temperature=0.7,
    stream=stream
)

if stream:
    for chunk in response:
        delta = chunk.choices[0].delta

        if delta.content:
            print(delta.content, end="", flush=True)
else:
    print(response.choices[0].message.content)