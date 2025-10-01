import os
import pandas as pd

from rewriter import rewrite_file
dashed_line = "----------------------------------------------------------"
tagged_line = "##########################################################"

def list_subfolders(path):
    """
    Returns a list of subfolders for the given path.

    :param path: The directory path to search
    :return: A list of subfolder paths
    """
    try:
        subfolders = [os.path.join(path, name) for name in os.listdir(path)
                      if os.path.isdir(os.path.join(path, name))]
        return subfolders
    except FileNotFoundError:
        print(f"The path '{path}' does not exist.")
        return []
    except PermissionError:
        print(f"Permission denied accessing '{path}'.")
        return []


def skrubify_pipeline(path):
    print(dashed_line)
    print(path)
    src_path = path + '/aide_pipeline.py'
    result_path = path + '/working/'
    with open(src_path, "r") as f:
        source_code = f.read()
    skrub_code = rewrite_file(src_path)

    try:
        exec(source_code)
        exec(skrub_code)
        original_submission = pd.read_csv(result_path + "/submission.csv")
        skrub_submission = pd.read_csv(result_path + "/submission_skrub.csv")
        if original_submission.equals(skrub_submission):
            print("Submissions are equal. No errors found.")
        else:
            print("Submissions are not equal.")
            print(dashed_line)
            print(original_submission)
            print(dashed_line)
            print(skrub_submission)
            print(dashed_line)

    except Exception as e:
        print(dashed_line)
        print("An error occurred")
        print(e)
    print(tagged_line)


if __name__ == "__main__":
    folder_path = "pipelines/"
    subfolders = list_subfolders(folder_path)
    print(tagged_line)
    for subfolder in subfolders:
        skrubify_pipeline(subfolder)
