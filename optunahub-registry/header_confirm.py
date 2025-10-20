import os

import frontmatter


def header_confirm(path: str) -> None:
    post = frontmatter.load(path)

    assert "author" in post.keys(), f"author is not found in {path}"
    assert "title" in post.keys(), f"title is not found in {path}"
    assert "description" in post.keys(), f"description is not found in {path}"
    assert "tags" in post.keys(), f"tags is not found in {path}"
    assert "optuna_versions" in post.keys(), f"optuna_versions is not found in {path}"
    assert "license" in post.keys(), f"license is not found in {path}"
    assert post["author"] != "", f"author is empty in {path}"
    assert post["title"] != "", f"title is empty in {path}"
    assert post["description"] != "", f"description is empty in {path}"
    assert post["tags"] != "", f"tags is empty in {path}"
    assert post["optuna_versions"] != "", f"optuna_versions is empty in {path}"
    assert post["license"] != "", f"license is empty in {path}"


if __name__ == "__main__":
    # Check all README files under the `package` directory.
    for root, dirs, files in os.walk("package"):
        for file in files:
            if file == "README.md":
                header_confirm(os.path.join(root, file))

    # Check the template file.
    header_confirm("template/README.md")
