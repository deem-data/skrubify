import os
import sys
from openai import OpenAI
from importlib import resources
import time

prompts = [
    "skrub_rewriter_prompt.txt",
    "skrub_rewriter_prompt_examples.txt",
    "skrub_rewriter_few_shot.txt",
    "skrub_rewriter_prompt_v2.txt",
]


def rm_white_space(text: str) -> str:
    """Remove trailing whitespace from every line in the text."""
    tmp = "\n".join(line.rstrip() for line in text.splitlines())
    return tmp


SYSTEM_PROMPTS = [
    rm_white_space(
        resources.files("skrubify.prompt_templates")
        .joinpath(prompt)
        .read_text(encoding="utf-8")
    )
    for prompt in prompts
]


def rewrite_file(file_path: str, mode: int, model: str = "gpt-4.1") -> str:
    """Read a file, append to SYSTEM_PROMPT, and rewrite with OpenAI."""
    with open(file_path, "r") as f:
        source_code = f.read()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[mode]},
            {"role": "user", "content": f"Rewrite this pipeline:\n\n```python\n{source_code}\n```"},
        ],
        temperature=0,
    )
    t1 = time.time()
    print(f"LLM inference time: {t1 - t0:.2f} seconds")

    return response.choices[0].message.content


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rewrite_prototype.py <file_to_rewrite.py> [output_file.py]")
        sys.exit(1)

    model = "gpt-4.1"
    prompt_mode = int(sys.argv[1])
    input_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    else:
        path_elements = input_file.split("/")
        path_elements[-1] = "{}_skrubified_prompt{}_{}.py".format(path_elements[-1][:-3], prompt_mode, model)
        output_file = "/".join(path_elements)
    rewritten_code = rewrite_file(input_file, mode=prompt_mode, model=model)

    if output_file:
        with open(output_file, "w") as f:
            f.write(rewritten_code)
        print(f"Rewritten pipeline saved to {output_file}")
    else:
        print(rewritten_code)
