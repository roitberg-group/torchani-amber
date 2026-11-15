r"""Generate include/torchani.h and interface/torchani.F90 from interface.yaml"""

from pathlib import Path

import yaml
import jinja2

INTERFACE_DIR = Path(__file__).resolve().parent
REPO_DIR = INTERFACE_DIR.parent

TEMPLATE_TO_OUTPUT = {
    "torchani.h.jinja": REPO_DIR / "include" / "torchani.h",
    "torchani.F90.jinja": INTERFACE_DIR / "torchani.F90",
}

C_TYPES = {
    "int": "int",
    "double": "double",
    "bool": "bool",
    "str": "const char*",
}

FORTRAN_TYPES = {
    "int": "integer(c_int)",
    "double": "real(c_double)",
    "bool": "logical(c_bool)",
    "str": "character(len=1, kind=c_char)",
}


def c_declaration(arg: dict) -> str:
    ctype = C_TYPES[arg["dtype"]]
    name = arg["name"]
    shape = arg["shape"]
    if not shape:
        if arg["dtype"] == "str" or arg["intent"] == "in":
            return f"{ctype} {name}"
        return f"{ctype}* {name}"
    trailing_dims = shape[1:]
    if all(isinstance(d, int) for d in trailing_dims):
        suffix = "".join(f"[{d}]" for d in trailing_dims)
        return f"{ctype} {name}[]{suffix}"
    return f"{ctype}* {name}"


def fortran_declaration(arg: dict) -> str:
    ftype = FORTRAN_TYPES[arg["dtype"]]
    name = arg["name"]
    intent = arg["intent"]
    if arg["shape"] or arg["dtype"] == "str":
        return f"{ftype}, intent({intent}) :: {name}(*)"
    if intent == "in":
        return f"{ftype}, value, intent(in) :: {name}"
    return f"{ftype}, intent({intent}) :: {name}"


def prepare_arg(arg: dict) -> dict:
    arg = {"intent": "in", "shape": [], **arg}
    if arg["dtype"] in ("bool", "str"):
        if arg["shape"] or arg["intent"] != "in":
            raise ValueError(f"{arg['name']}: bool and str args must be scalar inputs")
    arg["c_decl"] = c_declaration(arg)
    arg["f_binding_decl"] = fortran_declaration(arg)
    arg["f_wrapper_decl"] = arg["f_binding_decl"]
    arg["f_call_expr"] = arg["name"]
    if arg["dtype"] == "bool":
        arg["f_wrapper_decl"] = f"logical, intent(in) :: {arg['name']}"
        arg["f_call_expr"] = f"fbool_to_cbool({arg['name']})"
    return arg


def prepare_function(fn: dict) -> dict:
    fn = {"comment": "", "fortran_wrapper": "auto", "c_dummy_definition": False, **fn}
    fn["args"] = [prepare_arg(arg) for arg in fn["args"]]
    dtypes = {arg["dtype"] for arg in fn["args"]}
    needs_wrapper = bool(dtypes & {"bool", "str"})
    if "str" in dtypes and fn["fortran_wrapper"] != "manual":
        raise ValueError(f"{fn['name']}: str args require 'fortran_wrapper: manual'")
    fn["comment_lines"] = fn["comment"].splitlines()
    fn["generate_f_wrapper"] = needs_wrapper and fn["fortran_wrapper"] != "manual"
    fn["f_binding_name"] = fn["name"]
    if needs_wrapper:
        fn["f_binding_name"] = "internal_" + fn["name"].removeprefix("torchani_")
    return fn


def main() -> None:
    spec = yaml.safe_load((INTERFACE_DIR / "interface.yaml").read_text())
    functions = [prepare_function(fn) for fn in spec["functions"]]
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(INTERFACE_DIR / "templates"),
        undefined=jinja2.StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    for template_name, output_path in TEMPLATE_TO_OUTPUT.items():
        text = env.get_template(template_name).render(functions=functions)
        if not output_path.exists() or output_path.read_text() != text:
            output_path.write_text(text)
            print(f"Regenerated {output_path.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()
