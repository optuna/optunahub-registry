import os

import frontmatter


def header_confirm(path: str) -> None:
    post = frontmatter.load(path)
    assert "author" in post.keys(), f"author is not found in {path}"
    assert isinstance(post["author"], str), f"author is not a string in {path}"
    assert "title" in post.keys(), f"title is not found in {path}"
    assert isinstance(post["title"], str), f"title is not a string in {path}"
    assert "description" in post.keys(), f"description is not found in {path}"
    assert isinstance(post["description"], str), f"description is not a string in {path}"
    assert "tags" in post.keys(), f"tags is not found in {path}"
    assert isinstance(post["tags"], list), f"tags is not a list in {path}"
    assert all(
        isinstance(v, str) for v in post["tags"]
    ), f"tags is not a list of strings in {path}"
    assert "optuna_versions" in post.keys(), f"optuna_versions is not found in {path}"
    assert isinstance(post["optuna_versions"], list), f"optuna_versions is not a list in {path}"
    assert all(
        isinstance(v, str) for v in post["optuna_versions"]
    ), f"optuna_versions is not a list of strings in {path}"
    assert "license" in post.keys(), f"license is not found in {path}"
    assert isinstance(post["license"], str), f"license is not a string in {path}"
    assert post["license"] == "MIT License", f"license must be 'MIT License' in {path}"


if __name__ == "__main__":
    # Check all README files under the `package` directory.
    for root, dirs, files in os.walk("package"):
        for file in files:
            if file == "README.md":
                header_confirm(os.path.join(root, file))

    # Check the template file.
    header_confirm("template/README.md")
