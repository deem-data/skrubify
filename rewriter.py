import os
import argparse
from skrubify import Skrubify
default_prompt_mode = 5

def main():
    parser = argparse.ArgumentParser(
        description="Rewrite a Python file with a chosen model and prompt mode."
    )
    parser.add_argument(
        "input_file",
        help="Path to the Python file to rewrite."
    )
    parser.add_argument(
        "-p", "--prompt-mode",
        type=int,
        default=default_prompt_mode,
        help="Prompt mode (integer)."
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to save the rewritten file. Defaults to input file name with suffix."
    )
    parser.add_argument(
        "-m", "--model",
        default="gpt-4.1",
        help="Model to use (default: gpt-4.1)."
    )
    parser.add_argument(
        "-d", "--dry-run",
        action="store_true",
        help="Show what would be done without modifying or writing files."
    )

    # Determine output file if not given
    args = parser.parse_args()
    if args.output:
        output_file = args.output
    else:
        base, ext = os.path.splitext(args.input_file)
        output_file = f"{base}_skrubified_prompt{args.prompt_mode}_{args.model}{ext}"

    print()
    print(f"Input file: {args.input_file}")
    print(f"Prompt mode: {args.prompt_mode}")
    print(f"Model: {args.model}")
    print(f"Output file: {output_file}")

    if args.dry_run:
        print("\n[Dry Run] No files will be written or modified.")
    else:
        with open(args.input_file, "r") as f:
            source_code = f.read()

        rewriter = Skrubify()
        rewritten_code = rewriter.rewrite(source_code, mode=args.prompt_mode, model=args.model)

        if output_file:
            with open(output_file, "w") as f:
                f.write(rewritten_code)
            print(f"Rewritten pipeline saved to {output_file}")
        else:
            print(rewritten_code)


if __name__ == "__main__":
    main()