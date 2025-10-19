from setuptools import setup, find_packages

setup(
    name="skrubify",
    version="0.1.0",
    description="Data cleaning toolkit",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "openai", "google-genai", "python-dotenv"
    ],
)
