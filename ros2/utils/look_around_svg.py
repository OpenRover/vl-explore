from sys import stdin, stdout
from .look_around import RenderContext, LookAroundDatabase

if __name__ == "__main__":
    db: LookAroundDatabase
    for line in stdin:
        line = line.strip()
        if line == "::EOF::":
            break
        elif not line or line.startswith("#"):
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
            ctx.fig.savefig(stdout, transparent=True, format="svg")
            ctx.fig.savefig("output.svg", transparent=True, format="svg")
            ctx.fig.savefig("output.pdf", transparent=True, format="pdf")
