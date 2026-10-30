"""Generate portable, text-preserving SVG illustrations for the project README."""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


def document(width, height, title, body):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">
<title id="title">{escape(title)}</title>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>
<style>text{{font-family:"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",Arial,sans-serif;fill:#172b4d}} .muted{{fill:#52657a}} .edge{{fill:none;stroke:#64748b;stroke-width:2.5;marker-end:url(#arrow)}} </style>
<rect width="100%" height="100%" fill="white"/>
{body}
</svg>\n'''


def text(x, y, value, size=22, bold=False, cls="", anchor="middle"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{700 if bold else 400}" class="{cls}">{escape(value)}</text>'


def box(x, y, w, h, fill="#f1f5f9", stroke="#b7c8d9", radius=12):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'


def edge(points):
    return f'<path d="{points}" class="edge"/>'


def architecture():
    parts = [text(700, 53, "SARP 系统架构", 34, True),
             text(700, 85, "Structure-Aware Reasoning Prototype", 22, cls="muted"),
             box(40, 115, 1320, 95, "#edf4fc"),
             text(700, 150, "任务输入", 26, True),
             text(700, 187, "表格公式 · 图像流水线 · 动态图维护 · 逻辑查询训练 · 视频张量去噪", 23)]
    parts += [edge("M 460 210 L 460 245"), edge("M 1140 210 L 1140 245"),
              box(40, 250, 840, 110, "#eef7f3", "#9bc5b2"),
              text(460, 283, "声明式接入：Adapter.build()", 25, True),
              text(460, 317, "Graph + Operation", 23),
              text(460, 344, "输入、节点、依赖、算子及复用规则", 21, cls="muted"),
              box(920, 250, 440, 110, "#fff5e9", "#d7b98e"),
              text(1140, 283, "执行后端接入", 25, True),
              text(1140, 317, "Task + execute()", 23),
              text(1140, 344, "有环更新、训练及预算语义", 21, cls="muted"),
              edge("M 460 360 L 460 400"),
              box(40, 405, 840, 105, "#eef7f3", "#9bc5b2"),
              text(460, 440, "计算图准备", 25, True),
              text(460, 473, "依赖排序 · 内容指纹 · 变化传播 · 输出依赖分析", 22),
              text(460, 499, "Session 保存有效结果与版本状态", 20, cls="muted"),
              edge("M 460 510 L 460 550"), edge("M 1140 360 L 1140 550"),
              box(40, 555, 1320, 145, "#edf4fc"),
              text(700, 591, "方案规划：Planner", 26, True),
              text(700, 620, "按任务声明的能力生成候选方案", 21, cls="muted")]
    for x, label, fill, stroke in [(180, "范围收缩", "#e2f0e8", "#9bc5b2"),
                                  (570, "节点合并", "#eee7f6", "#b8a5d1"),
                                  (960, "预算分配", "#fff0dc", "#d7b98e")]:
        parts += [box(x, 638, 260, 42, fill, stroke, 8), text(x + 130, 667, label, 23, True)]
    parts += [edge("M 460 700 L 460 760"), edge("M 1140 700 L 1140 760"),
              text(700, 734, "合法性检查 → 代价比较 → 方案选择", 22, cls="muted"),
              box(40, 765, 840, 100, "#eef7f3", "#9bc5b2"),
              text(460, 802, "计算图执行器", 26, True),
              text(460, 838, "按执行范围和节点分组调用算子，维护缓存", 22),
              box(920, 765, 440, 100, "#fff5e9", "#d7b98e"),
              text(1140, 802, "领域执行后端", 26, True),
              text(1140, 838, "处理任务特定的计算语义", 22),
              edge("M 460 865 L 460 910"), edge("M 1140 865 L 1140 910"),
              box(40, 915, 1320, 100, "#edf4fc"),
              text(700, 953, "结果与执行记录", 26, True),
              text(700, 988, "输出结果 · 算子计数 · 准备 / 规划 / 执行耗时 · 验证记录", 23),
              text(700, 1053, "声明式接入自动处理精确 DAG；预算方案和任务特定约束由相应后端提供。", 21, cls="muted")]
    (OUT / "architecture.svg").write_text(document(1400, 1080, "SARP 系统架构", "\n".join(parts)))


def transformations():
    parts = [text(700, 49, "执行变换：范围收缩与节点合并", 32, True),
             text(700, 84, "示例：x 更新，y 不变；保留所需输出，减少重算与重复计算。", 22, cls="muted")]
    for panel, x in enumerate((30, 500, 970)):
        parts.append(box(x, 120, 400, 415, "#fafcfe"))
        parts.append(text(x + 200, 159, ("原始计算图", "确定重算范围", "合并相同表达式")[panel], 25, True))
        for path in (f"M {x+80} 278 L {x+135} 233", f"M {x+80} 278 L {x+135} 333") if panel < 2 else (f"M {x+80} 278 L {x+135} 278",):
            parts.append(edge(path))
        if panel < 2:
            parts += [edge(f"M {x+235} 233 L {x+280} 233"), edge(f"M {x+235} 333 L {x+280} 333")]
        else:
            parts += [edge(f"M {x+235} 278 L {x+280} 233"), edge(f"M {x+235} 278 L {x+280} 333")]
        parts.append(edge(f"M {x+80} 438 L {x+135} 438"))
        fill = "#e2f0e8" if panel == 1 else "#edf4fc"
        parts += [box(x+20, 255, 60, 46, "#ffe6df", "#dcad9f", 8), text(x+50, 286, "x", 24, True)]
        for y in ((210, 310) if panel < 2 else (255,)):
            parts += [box(x+135, y, 100, 46, "#eee7f6" if panel == 2 else fill, radius=8),
                      text(x+185, y+31, "f(x)", 23)]
        for y, name in ((210, "g"), (310, "h")):
            parts += [box(x+280, y, 90, 46, fill, radius=8), text(x+325, y+31, name, 23)]
        gray = "#edf0f4" if panel else "#edf4fc"
        parts += [box(x+20, 415, 60, 46, gray, radius=8), text(x+50, 446, "y", 24),
                  box(x+135, 415, 100, 46, gray, radius=8), text(x+185, 446, "k(y)", 23)]
        if panel:
            parts.append(text(x+315, 445, "复用缓存", 21, cls="muted"))
        parts.append(text(x+200, 507, ("两个分支重复计算 f(x)", "只重算受 x 变化影响的节点", "f(x) 执行一次，供 g、h 使用")[panel], 20, cls="muted"))
    parts += [edge("M 438 320 L 487 320"), edge("M 908 320 L 957 320"),
              text(700, 580, "范围决策确定“算哪些”，分组决策确定“哪些计算可以只做一次”。", 23, cls="muted")]
    (OUT / "transformations.svg").write_text(document(1400, 610, "SARP 范围收缩与节点合并示意", "\n".join(parts)))


if __name__ == "__main__":
    architecture()
    transformations()
    print(OUT)
