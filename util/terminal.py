# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
def confirm(question: str, auto_rej: bool = False, auto_acc: bool=False) -> bool:
    dfl = ""

    if auto_rej:
        assert not auto_acc, "Cannot auto accept and reject at the same time"
        dfl = "(n) "

    if auto_acc:
        assert not auto_rej, "Cannot auto accept and reject at the same time"
        dfl = "(Y) "

    prompt = f"{question} [Y/n] {dfl}"

    while True:
        match input(prompt):
            case "Y":
                return True
            case "n":
                return False
            case "":
                if auto_rej:
                    return False
                if auto_acc:
                    return True
        print(f"Please respond with 'Y' or 'n'")
