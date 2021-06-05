from setuptools import setup, find_namespace_packages

setup(name='unsupervised-metrics',
    version='0.1',
    description='Self-Learning for Unsupervised Metrics',
    keywords = ["Unsupervised", "Metrics", "Quality Estimation", "Machine Translation", "NLP", "Deep Learning"],
    url='https://github.com/potamides/XMoverAlign',
    author='Jonas Belouadi',
    author_email='jonasjohannesfranz.belouadi@stud.tu-darmstadt.de',
    packages=find_namespace_packages(include=["metrics*"]),
    install_requires=[
        "faiss-gpu==1.6.5",
        "pyemd==0.5.1",
        "torch==1.7.1",
        "transformers==4.5.1",
        "datasets==1.6.1",
        "cupy-cuda100==8.5.0",
        "nltk==3.5",
        "sentencepiece==0.1.95",
        "mosestokenizer==1.1.0",
        "simalign @ https://github.com/cisnlp/simalign/archive/refs/tags/v0.2.zip",
        "truecase==0.0.12",
        "tabulate==0.8.9"
    ],
    python_requires=">=3.9.0",
    zip_safe=False,
)
