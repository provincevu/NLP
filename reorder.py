import re

with open("understand-dashboard.html", "r") as f:
    content = f.read()

# Define the section markers
markers = [
    "<!-- HERO -->",
    "<!-- STATS STRIP -->",
    "<!-- PROBLEM & SOLUTION -->",
    "<!-- HOW IT WORKS -->",
    "<!-- PIPELINE -->",
    "<!-- BASELINES EVALUATION -->",
    "<!-- FILE EXPLORER -->",
    "<!-- ARCHITECTURE -->",
    "<!-- DATA FLOW -->",
    "<!-- CORE ALGORITHMS -->",
    "<!-- TECH STACK -->",
    "<!-- DESIGN DECISIONS -->",
    "<!-- FOOTER -->"
]

# We need to split the content based on these markers.
# Let's find the indices of these markers.
indices = []
for m in markers:
    idx = content.find(m)
    if idx != -1:
        indices.append((idx, m))
    else:
        print(f"Not found: {m}")

indices.sort()

# Also we need to get the header (before HERO)
header = content[:indices[0][0]]

sections = {}
for i in range(len(indices)):
    start = indices[i][0]
    end = indices[i+1][0] if i + 1 < len(indices) else len(content)
    # Be careful not to include FOOTER if there is stuff after it, but FOOTER is the last section.
    # Actually, FOOTER might include script tags. We will just capture until the end of the file.
    m = indices[i][1]
    sections[m] = content[start:end]

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

print("Done generating reordered HTML")
