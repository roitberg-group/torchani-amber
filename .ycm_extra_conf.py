# type: ignore
import os
from pathlib import Path

database = None


def FindCorrespondingSourceFile(filename):
    extension = os.path.splitext(filename)[1]
    if extension in [".h", ".hxx", ".hpp", ".hh"]:
        basename = os.path.splitext(filename)[0]
        for extension in [".cpp", ".cxx", ".cc", ".c", ".m", ".mm"]:
            replacement_file = basename + extension
            if os.path.exists(replacement_file):
                return replacement_file
    return filename


def Settings(**kwargs):
    # Do NOT import ycm_core at module scope.
    import ycm_core

    global database
    compilation_database_folder = Path(__file__).parent / "build"
    if database is None and compilation_database_folder.is_dir():
        database = ycm_core.CompilationDatabase(str(compilation_database_folder))

    language = kwargs["language"]

    if language == "cfamily":
        # If the file is a header, try to find the corresponding source file and
        # retrieve its flags from the compilation database if using one. This is
        # necessary since compilation databases don't have entries for header files.
        # In addition, use this source file as the translation unit. This makes it
        # possible to jump from a declaration in the header file to its definition
        # in the corresponding source file.
        filename = FindCorrespondingSourceFile(kwargs["filename"])

        if not database:
            return {}
            # return {
            # "flags": [],
            # "include_paths_relative_to_dir": str(Path(__file__).parent),
            # "override_filename": filename,
            # }

        compilation_info = database.GetCompilationInfoForFile(filename)
        if not compilation_info.compiler_flags_:
            return {}

        # Bear in mind that compilation_info.compiler_flags_ does NOT return a
        # python list, but a "list-like" StringVec object.
        final_flags = list(compilation_info.compiler_flags_) + ["-stdlib=libc++"]

        return {
            "flags": final_flags,
            "include_paths_relative_to_dir": compilation_info.compiler_working_dir_,
            "override_filename": filename,
        }

    return {}
