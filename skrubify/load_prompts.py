from importlib import resources

prompts = [
    "skrub_rewriter_prompt.txt",
    "skrub_rewriter_prompt_examples.txt",
    "skrub_rewriter_few_shot.txt",
    "skrub_rewriter_prompt_v2.txt",
    "skrub_rewriter_few_shot_with_custom_concat.txt",
    "skrub_rewriter_few_shot_with_concat_op.txt",
    "skrub_rewriter_few_shot_only_skrub.txt"
]


def rm_white_space(text: str) -> str:
    """Remove trailing whitespace from every line in the text."""
    tmp = "\n".join(line.rstrip() for line in text.splitlines())
    return tmp


def load_prompts():
    return [
        rm_white_space(
            resources.files("skrubify.prompt_templates")
            .joinpath(prompt)
            .read_text(encoding="utf-8")
        )
        for prompt in prompts
    ]

