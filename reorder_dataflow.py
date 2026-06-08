import os

with open("understand-dashboard.html", "r") as f:
    content = f.read()

markers = [
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
    "<!-- DESIGN DECISIONS -->",
    "<!-- FOOTER -->"
]

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

# Desired order: Move DATA FLOW before BASELINES EVALUATION
desired_order = [
    "<!-- HERO -->",
    "<!-- STATS STRIP -->",
    "<!-- PROBLEM & SOLUTION -->",
    "<!-- HOW IT WORKS -->",
    "<!-- ARCHITECTURE -->",
    "<!-- PIPELINE -->",
    "<!-- CORE ALGORITHMS -->",
    "<!-- DATA FLOW -->",
    "<!-- BASELINES EVALUATION -->",
    "<!-- TECH STACK -->",
    "<!-- DESIGN DECISIONS -->",
    "<!-- FOOTER -->"
]

new_content = header
for m in desired_order:
    if m in sections:
        new_content += sections[m]
    else:
        print(f"Warning: {m} not in sections")

with open("understand-dashboard.html", "w") as f:
    f.write(new_content)

print("Done reordering HTML.")
