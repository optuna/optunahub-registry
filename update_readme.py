from __future__ import annotations

from enum import Enum
from enum import unique
import glob
import os

import frontmatter
import mdutils


@unique
class HubCategory(Enum):
    SAMPLER = "samplers"
    VISUALIZATION = "visualization"


def main() -> None:
    mdFile = mdutils.MdUtils(file_name="README", title="OptunaHub Registry")
    mdFile.new_header(level=1, title="List of Packages")
    for category in HubCategory:
        mdFile.new_header(level=2, title=category.value.title())
        package_paths = glob.glob(f"package/{category.value}/**")
        packages = ["Title", "Description"]
        columns = len(packages)
        for p in package_paths:
            fm = frontmatter.load(f"{p}/README.md")
            thumbnail = f"{p}/img/thumbnail.png"
            if os.path.exists(thumbnail):
                description = f'<img src="{thumbnail}" width="160px" height="120px" alt="{fm["description"]}">'
            else:
                description = fm["description"]
            packages.extend(
                [
                    mdutils.tools.TextUtils.text_external_link(fm["title"], p),
                    description,
                ]
            )
        mdFile.new_table(
            columns=columns,
            rows=1 + len(package_paths),
            text=packages,
            text_align="center",
        )
    mdFile.create_md_file()


if __name__ == "__main__":
    main()
