import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME = "Perceptron"
USER_NAME = "Aayush007A"

setuptools.setup(
    name=f"{PROJECT_NAME}-{USER_NAME}",
    version="0.0.1",
    author=USER_NAME,
    author_email="upadhyaychirag851@gmail.com",
    description="A simple perceptron package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USER_NAME}/{PROJECT_NAME}",
    project_urls={
        "Bug Tracker":f"https://github.com/{USER_NAME}/{PROJECT_NAME}/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src"),
    python_requires =">=3.10", 
    install_requires=[
        "numpy==1.25.2","pandas==1.5.3","joblib==1.2.0"
    ]
)