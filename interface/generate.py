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
    "write_line_fn": "mlip_write_line_fn",
}

FORTRAN_TYPES = {
    "int": "integer(c_int)",
    "double": "real(c_double)",
    "bool": "logical(c_bool)",
    "str": "character(len=1, kind=c_char)",
    "write_line_fn": "type(c_funptr)",
}


def c_declaration(arg: dict) -> str:
    ctype = C_TYPES[arg["dtype"]]
    name = arg["name"]
    shape = arg["shape"]
    if not shape:
        if arg["dtype"] == "str" and arg["intent"] != "in":
            return f"char {name}[]"
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
        if arg["shape"] or arg["intent"] == "inout":
            raise ValueError(
                f"{arg['name']}: bool and str args must be scalar, in or out"
            )
    name = arg["name"]
    c_name = f"c_{name}"
    arg["c_decl"] = c_declaration(arg)
    arg["f_binding_decl"] = fortran_declaration(arg)
    # Fortran wrapper: dummy arg declaration, locals, statements before and after the
    # call to the bind(c) interface, and the expression passed to it
    arg["f_wrapper_decl"] = arg["f_binding_decl"]
    arg["f_local_decls"] = []
    arg["f_pre_call"] = []
    arg["f_post_call"] = []
    arg["f_call_expr"] = name
    arg["f_is_wrapper_arg"] = True
    if arg["dtype"] == "bool" and arg["intent"] == "in":
        arg["f_wrapper_decl"] = f"logical, intent(in) :: {name}"
        arg["f_call_expr"] = f"fbool_to_cbool({name})"
    elif arg["dtype"] == "bool":
        arg["f_wrapper_decl"] = f"logical, intent(out) :: {name}"
        arg["f_local_decls"] = [f"logical(c_bool) :: {c_name}"]
        arg["f_post_call"] = [f"{name} = {c_name}"]
        arg["f_call_expr"] = c_name
    elif arg["dtype"] == "str" and arg["intent"] == "in":
        arg[
            "f_wrapper_decl"
        ] = f'character(len=*, kind=kind("a")), intent(in) :: {name}'
        arg["f_local_decls"] = [
            f"character(len=len_trim({name}) + 1, kind=c_char) :: {c_name}"
        ]
        arg["f_pre_call"] = [f"{c_name} = trim({name}) // c_null_char"]
        arg["f_call_expr"] = c_name
    elif arg["dtype"] == "str":
        # Copied up to the null char, and padded with blanks
        idx = f"{name}_idx"
        arg[
            "f_wrapper_decl"
        ] = f'character(len=*, kind=kind("a")), intent(out) :: {name}'
        arg["f_local_decls"] = [
            f"character(len=1, kind=c_char) :: {c_name}(len({name}) + 1)",
            f"integer :: {idx}",
        ]
        arg["f_post_call"] = [
            f'{name} = ""',
            f"do {idx} = 1, len({name})",
            f"    if ({c_name}({idx}) == c_null_char) exit",
            f"    {name}({idx}:{idx}) = {c_name}({idx})",
            "enddo",
        ]
        arg["f_call_expr"] = c_name
    return arg


def outputs_start(args: list[dict]) -> int:
    start = len(args)
    while start > 0 and args[start - 1]["intent"] != "in":
        start -= 1
    return start


def prepare_function(fn: dict) -> dict:
    fn = {"comment": "", "fortran_wrapper": "auto", "fortran_generic": "", **fn}
    fn["args"] = [prepare_arg(arg) for arg in fn["args"]]
    args_by_name = {arg["name"]: arg for arg in fn["args"]}
    for arg in fn["args"]:
        if arg["dtype"] != "str" or arg["intent"] != "out":
            continue
        len_name = f"{arg['name']}_len"
        if len_name not in args_by_name or args_by_name[len_name]["dtype"] != "int":
            raise ValueError(f"{arg['name']}: str outputs need an int arg {len_name}")
        # The buffer length is taken from the Fortran string
        args_by_name[len_name]["f_is_wrapper_arg"] = False
        args_by_name[len_name]["f_call_expr"] = f"size(c_{arg['name']})"
    dtypes = {arg["dtype"] for arg in fn["args"]}
    fn["comment_lines"] = fn["comment"].splitlines()
    fn["has_bool_args"] = "bool" in dtypes
    fn["has_str_args"] = "str" in dtypes
    if fn["fortran_wrapper"] not in ("auto", "manual", "none"):
        raise ValueError(
            f"{fn['name']}: unknown fortran_wrapper {fn['fortran_wrapper']}"
        )
    needs_wrapper = fn["has_bool_args"] or fn["has_str_args"]
    fn["generate_f_wrapper"] = needs_wrapper and fn["fortran_wrapper"] == "auto"
    fn["f_wrapper_args"] = [arg for arg in fn["args"] if arg["f_is_wrapper_arg"]]
    fn["outputs_start"] = outputs_start(fn["args"])
    fn["f_wrapper_outputs_start"] = outputs_start(fn["f_wrapper_args"])
    fn["f_binding_name"] = fn["name"]
    if needs_wrapper or fn["fortran_wrapper"] != "auto":
        fn["f_binding_name"] = "internal_" + fn["name"].removeprefix("mlip_")
    return fn


def main() -> None:
    spec = yaml.safe_load((INTERFACE_DIR / "interface.yaml").read_text())
    functions = [prepare_function(fn) for fn in spec["functions"]]
    fortran_generics: dict[str, list[str]] = {}
    for fn in functions:
        if fn["fortran_generic"]:
            fortran_generics.setdefault(fn["fortran_generic"], []).append(fn["name"])
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(INTERFACE_DIR / "templates"),
        undefined=jinja2.StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    for template_name, output_path in TEMPLATE_TO_OUTPUT.items():
        text = env.get_template(template_name).render(
            functions=functions, fortran_generics=fortran_generics
        )
        if not output_path.exists() or output_path.read_text() != text:
            output_path.write_text(text)
            print(f"Regenerated {output_path.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()
