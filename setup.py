from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup(
  name = 'SSL_graph_audio',
  packages = find_packages(exclude=[]),
  version = '0.1',
  license='MIT',
  description = 'Graph SSL task Library for PyTorch Geometric',
  long_description_content_type="text/markdown",
  long_description=README,
  author = 'Amir Shirian',
  author_email = 'mail.amirdonte15@gmail.com',
  url = 'https://github.com/AmirSh15/SSL_graph_audio',
  keywords = [
    'machine learning',
    'graph deep learning',
    'self-supervised learning',
  ],
  install_requires=[
    'torch>=1.10',
    'torch_geometric>=2.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
