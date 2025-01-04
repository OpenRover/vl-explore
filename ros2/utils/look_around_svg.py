import sys, io
from .look_around import RenderContext, LookAroundDatabase
import matplotlib.pyplot as plt

def dump(fig: plt.Figure):
    # Create an in-memory string buffer
    svg_buffer = io.StringIO()
    # Save the figure as SVG into the buffer
    fig.savefig(svg_buffer, transparent=True, format='svg')
    svg_data = svg_buffer.getvalue()
    svg_buffer.close()
    print(svg_data)

if __name__ == "__main__":
    db: LookAroundDatabase
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        elif line.startswith("=" * 3):
            name = line.strip("=")
            db = LookAroundDatabase(name)
            continue
        else:
            db.add(line)
    with RenderContext(theme="light") as ctx:
        db.render(ctx)
        with ctx.head_to(0):
            dump(ctx.fig)
