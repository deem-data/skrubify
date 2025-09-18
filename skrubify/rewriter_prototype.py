import os
import sys
from openai import OpenAI


SYSTEM_PROMPT = "EMPTY"
with open("prompt_templates/skrub_rewriter_prompt.txt","r") as f:
    prompt = f.read()


def rewrite_file(file_path: str, model: str = "gpt-4.1") -> str:
    """Read a file, append to SYSTEM_PROMPT, and rewrite with OpenAI."""
    with open(file_path, "r") as f:
        source_code = f.read()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Rewrite this pipeline:\n\n```python\n{source_code}\n```"},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rewrite_prototype.py <file_to_rewrite.py> [output_file.py]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    rewritten_code = rewrite_file(input_file)

    if output_file:
        with open(output_file, "w") as f:
            f.write(rewritten_code)
        print(f"Rewritten pipeline saved to {output_file}")
    else:
        print(rewritten_code)
