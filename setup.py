from pathlib import Path

import pkg_resources as pkg
from setuptools import find_packages, setup

FILE = Path(__file__).resolve()
REPO = FILE.parent
README = (REPO / "readme.md").read_text(encoding="utf-8")
REQUIREMENTS = [
    f"{x.name}{x.specifier}"  # type: ignore
    for x in pkg.parse_requirements((REPO / "requirements.txt").read_text())
]

setup(
    name="movie-subtitles",
    version="0.0.1",
    python_requires=">=3.10",
    description="Command line interface to transcribe movies",
    long_description=README,
    long_description_content_type="text/markdown",
    url="pip install git+https://github.com/mathiasesn/movie-subtitles.git.git",
    author="mathiasesn",
    author_email="mathiasesn1@gmail.com",
    packages=find_packages(include=["movie_subtitles", "movie_subtitles.*"]),
    entry_points={"console_scripts": ["movie-subtitles=movie_subtitles.cli:main"]},
    include_package_data=True,
    install_requires=REQUIREMENTS,
)
