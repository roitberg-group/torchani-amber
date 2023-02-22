set_python_command() {
    # This checks whether to use "python" or "python3" as a command
    # it tries to be smart by using python3 only if 'python' version is strictly
    # smaller than 'python3'.
    # this function sets the variable "python_command"
    unset plain_python_version
    unset python3_version
    unset python_command
    unset python_version
    unset python_dir
    command -v python 1>/dev/null 2>&1 || plain_python_version=0
    command -v python3 1>/dev/null 2>&1 || python3_version=0
    if [ -z ${CONDA_PREFIX} ]; then
        echo "WARNING: You seem to be running outside an Anaconda/Miniconda environment, this is not recommended"
    else
        unset PYTHONPATH
        echo "Conda environment ${CONDA_PREFIX} detected"
    fi

    if [ -z "${plain_python_version}" ]; then
        plain_python_version=$(python --version | awk '{print $2}')
        plain_python_version=${plain_python_version:0:3}
    fi

    if [ -z "${python3_version}" ]; then
        python3_version=$(python3 --version | awk '{print $2}')
        python3_version=${python3_version:0:3}
    fi

    if [ $(echo ${plain_python_version} '<' ${python3_version} | bc -l) -eq 1 ]; then
        python_command='python3'
        python_version=${python3_version}
        python_dir="$(which python3)"
    else
        python_command='python'
        python_version=${plain_python_version}
        python_dir="$(which python3)"
    fi

    echo "The most up to date python found seems to be"
    echo ${python_dir} "version" ${python_version} "called with comand" \"${python_command}\"
    echo "This python will be used."

    if [ $(echo ${python_version} '<' 3.6 | bc -l) -eq 1 ]; then
        echo "ERROR: Python version detected for ${python_dir} is ${python_version}"
        echo "The minimum acceptable version is 3.6, please update python"
    fi
    unset plain_python_version
    unset python3_version
    unset python_version
    unset python_dir
}
