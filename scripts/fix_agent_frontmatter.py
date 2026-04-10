import re
import sys
import glob
import os


def fix_frontmatter(content: str) -> str:
    match = re.search(r"^---[ \t]*\n(.*?)\n---[ \t]*\n", content, re.DOTALL)
    if not match:
        return content

    frontmatter = match.group(1)
    lines = frontmatter.split("\n")
    new_lines = []

    in_description = False
    already_block = False

    for line in lines:
        if line.startswith("description:"):
            in_description = True
            desc_text = line[len("description:") :].lstrip()

            if desc_text.startswith("|"):
                already_block = True
                new_lines.append(line)
            else:
                already_block = False
                new_lines.append("description: |")
                if desc_text:
                    new_lines.append(f"  {desc_text}")
            continue

        if in_description:
            if re.match(r"^(name|model|color|tools|version|author):", line):
                in_description = False
                new_lines.append(line)
            else:
                if line.strip() == "":
                    new_lines.append("")
                else:
                    if already_block:
                        new_lines.append(line)
                    else:
                        new_lines.append(f"  {line}")
        else:
            new_lines.append(line)

    new_frontmatter = "\n".join(new_lines)
    return content[: match.start()] + "---\n" + new_frontmatter + "\n---\n" + content[match.end() :]


def main():
    agent_dir = os.path.expanduser("~/.gemini/agents")
    md_files = glob.glob(os.path.join(agent_dir, "*.md"))

    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        fixed_content = fix_frontmatter(content)

        if fixed_content != content:
            print(f"Fixing {file_path}")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(fixed_content)
        else:
            print(f"Skipping {file_path} (no changes needed)")


if __name__ == "__main__":
    main()
