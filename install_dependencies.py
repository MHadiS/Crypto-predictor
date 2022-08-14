import subprocess
import platform
import os
import sys


SYSTEM = platform.system().lower()  # the user platform


def run(command):
    """Run a command and put the output into the installation.log

    Args:
        command (str): The command to run
    """
    command = command.split(' ')
    subprocess.call(command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)


def install(python_path, packages):
    """
    Installs Python packages.

    Args:
        python_path (str): The path to the python.
        dependencies (list): A list of dependency names.
    """
    for package in packages:
        print("\n", "\033[95m=" * 80, "\n")
        print(f"\033[93m Installing {package}... \033[1m")
        run(python_path + " -m pip install " + package)
        print(f"\033[92m {package} installed successfully. \033[1m")


def make_venv(name: str):
    """Make a virtual environment

    Args:
        name (str): name of the virtual environment
    """
    if SYSTEM == "windows":
        command = sys.executable + " -m venv " + name
    else:
        command = sys.executable + " -m venv " + name

    run(command)


def main():
    """Install the dependencies
    """
    dependencies = [
        "pandas",
        "numpy",
        "sklearn",
        "pickle5",
        "rich",
        "requests",
        "argparse",
        "yfinance",
        "pytest",
    ]
    want_venv = input("Do you want to install dependencies in a venv? [y/n]: ").lower()
    python_path = sys.executable
    if want_venv == "y":
        make_venv("dependencies")
        python_path = os.path.abspath("./dependencies")
        if SYSTEM == "windows":
            python_path += "\\Scripts\\python"
        else:
            python_path += "/bin/python3"

    install(python_path, dependencies)
    print("\n", "\033[95m=" * 80, "\n")
    print("\033[4m\033[96mAll done.\033[4m")
    print("\033[37m \033[0m")


if __name__ == "__main__":
    main()
