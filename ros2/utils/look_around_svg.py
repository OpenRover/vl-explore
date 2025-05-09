from sys import stdin, stdout, stderr
from matplotlib import font_manager, rc
from .look_around import RenderContext, LookAroundDatabase

if __name__ == "__main__":
    # /usr/share/fonts/truetype/msttcorefonts/times.ttf
    font_dirs = ["/usr/share/fonts/truetype/msttcorefonts"]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    rc("font", family="Times New Roman", size=12)
    rc("svg", fonttype="none")

    print("LookAround Rendering Started", file=stderr)
    db: LookAroundDatabase
    for line in stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        elif line.startswith("=" * 3):
            name = line.strip("=")
            db = LookAroundDatabase(name)
            continue
        else:
            db.add(line)
    print("LookAround Rendering Received", file=stderr)
    with RenderContext(theme="light") as ctx:
        db.render(ctx)
        with ctx.head_to(0):
            ctx.fig.savefig(stdout, transparent=True, format="svg")
            # ctx.fig.savefig("output.svg", transparent=True, format="svg")
            # ctx.fig.savefig("output.pdf", transparent=True, format="pdf")
    print("LookAround Rendering Complete", file=stderr)
