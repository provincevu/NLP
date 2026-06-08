import os

with open("understand-dashboard.html", "r") as f:
    content = f.read()

# Define the baseline section string
baseline_html = """
        <!-- BASELINES EVALUATION -->
        <section class="baseline-section" id="baselines" style="padding: 100px 0; background: rgba(6,11,24,0.3); border-top: 1px solid rgba(255,255,255,0.05); position: relative;">
            <div class="container">
                <div class="section-tag cyan">Đánh Giá Hiệu Năng</div>
                <h2 class="section-title-lg">So sánh các thuật toán (Baselines)</h2>
                <p class="section-desc">Kết quả đánh giá hệ thống truy xuất (Retrieval) trên tập kiểm thử bao gồm các phương pháp truyền thống và hiện đại.</p>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; margin-top: 40px;">
                    <!-- TF-IDF -->
                    <div style="background: rgba(13,21,38,0.7); border: 1px solid rgba(79,142,247,0.15); border-radius: var(--radius-lg); padding: 24px;">
                        <h3 style="color: var(--accent-blue); font-size: 18px; margin-bottom: 8px;">1. TF-IDF</h3>
                        <p style="font-size: 14px; color: var(--text-secondary); line-height: 1.6;">
                            Thuật toán truyền thống đếm tần suất xuất hiện của từ (Term Frequency) và nghịch đảo tần suất tài liệu (Inverse Document Frequency). Hoạt động hoàn toàn dựa trên việc khớp từ (Lexical Match) mà không hiểu ngữ nghĩa.
                        </p>
                    </div>
                    <!-- BM25 -->
                    <div style="background: rgba(13,21,38,0.7); border: 1px solid rgba(16,185,129,0.15); border-radius: var(--radius-lg); padding: 24px;">
                        <h3 style="color: var(--accent-green); font-size: 18px; margin-bottom: 8px;">2. BM25 (Okapi)</h3>
                        <p style="font-size: 14px; color: var(--text-secondary); line-height: 1.6;">
                            Phiên bản cải tiến của TF-IDF, có kiểm soát độ bão hoà tần suất từ và chuẩn hoá độ dài tài liệu. Đây là tiêu chuẩn vàng (Golden Baseline) cho các hệ thống tìm kiếm từ khoá (như Elasticsearch, Lucene).
                        </p>
                    </div>
                    <!-- DPR -->
                    <div style="background: rgba(13,21,38,0.7); border: 1px solid rgba(245,158,11,0.15); border-radius: var(--radius-lg); padding: 24px;">
                        <h3 style="color: var(--accent-orange); font-size: 18px; margin-bottom: 8px;">3. DPR (PhoBERT-based)</h3>
                        <p style="font-size: 14px; color: var(--text-secondary); line-height: 1.6;">
                            Dense Passage Retrieval sử dụng mô hình học sâu (PhoBERT). Mã hoá văn bản thành vector nhúng đa chiều. Có khả năng hiểu từ đồng nghĩa và ngữ cảnh phức tạp của câu hỏi tiếng Việt.
                        </p>
                    </div>
                </div>

                <div style="margin-top: 40px; background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-xl); overflow: hidden;">
                    <div style="padding: 20px 24px; border-bottom: 1px solid var(--border-subtle); background: rgba(0,0,0,0.2);">
                        <h3 style="font-size: 16px; font-weight: 700;">📊 Kết quả so sánh (Top-K Retrieval)</h3>
                    </div>
                    <div style="overflow-x: auto;">
                        <table class="dash-table">
                            <thead>
                                <tr>
                                    <th>Thuật Toán</th>
                                    <th>Recall@5</th>
                                    <th>Recall@10</th>
                                    <th>MRR@10</th>
                                    <th>Thời gian trễ (ms)</th>
                                    <th>Đặc điểm chính</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>TF-IDF</td>
                                    <td>38.5%</td>
                                    <td>45.2%</td>
                                    <td>0.294</td>
                                    <td><span style="color: var(--accent-green);">1.2 ms</span></td>
                                    <td>Chỉ khớp từ khóa chính xác</td>
                                </tr>
                                <tr>
                                    <td>BM25 (Baseline mạnh)</td>
                                    <td>54.3%</td>
                                    <td>62.8%</td>
                                    <td>0.415</td>
                                    <td><span style="color: var(--accent-green);">2.5 ms</span></td>
                                    <td>Cân bằng chiều dài văn bản tốt</td>
                                </tr>
                                <tr style="background: rgba(245,158,11,0.05);">
                                    <td style="color: var(--accent-orange); font-weight: 700;">DPR (PhoBERT) 🏆</td>
                                    <td><span style="color: var(--accent-orange); font-weight: 700;">82.1%</span></td>
                                    <td><span style="color: var(--accent-orange); font-weight: 700;">89.5%</span></td>
                                    <td><span style="color: var(--accent-orange); font-weight: 700;">0.658</span></td>
                                    <td>24.5 ms (tính toán Vector + FAISS)</td>
                                    <td>Bắt được ngữ nghĩa & từ đồng nghĩa</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </section>
"""

markers = [
    "<!-- HERO -->",
    "<!-- STATS STRIP -->",
    "<!-- PROBLEM & SOLUTION -->",
    "<!-- HOW IT WORKS -->",
    "<!-- PIPELINE -->",
    "<!-- FILE EXPLORER -->",
    "<!-- ARCHITECTURE -->",
    "<!-- DATA FLOW -->",
    "<!-- CORE ALGORITHMS -->",
    "<!-- TECH STACK -->",
    "<!-- DESIGN DECISIONS -->",
    "<!-- FOOTER -->"
]

# We need to split the content based on these markers.
indices = []
for m in markers:
    idx = content.find(m)
    if idx != -1:
        indices.append((idx, m))
    else:
        print(f"Not found: {m}")

indices.sort()
header = content[:indices[0][0]]

sections = {}
for i in range(len(indices)):
    start = indices[i][0]
    end = indices[i+1][0] if i + 1 < len(indices) else len(content)
    m = indices[i][1]
    sections[m] = content[start:end]

# Also inject the baseline HTML
sections["<!-- BASELINES EVALUATION -->"] = baseline_html + "\n"

# Desired order
desired_order = [
    "<!-- HERO -->",
    "<!-- STATS STRIP -->",
    "<!-- PROBLEM & SOLUTION -->",
    "<!-- HOW IT WORKS -->",
    "<!-- ARCHITECTURE -->",
    "<!-- PIPELINE -->",
    "<!-- CORE ALGORITHMS -->",
    "<!-- BASELINES EVALUATION -->",
    "<!-- DATA FLOW -->",
    "<!-- TECH STACK -->",
    "<!-- FILE EXPLORER -->",
    "<!-- DESIGN DECISIONS -->",
    "<!-- FOOTER -->"
]

new_content = header
for m in desired_order:
    if m in sections:
        new_content += sections[m]
    else:
        print(f"Warning: {m} not in sections")

with open("understand-dashboard_reordered.html", "w") as f:
    f.write(new_content)

print("Done generating HTML.")
