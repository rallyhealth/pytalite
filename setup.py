from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(name='pytalite',
      version='1.0.0',
      description='Python/Pyspark Package for Model-agnostic Evaluation and Diagnosis.',
      author='Jinghao Jia, Lun Yu',
      author_email='jiajinghao1998@gmail.com, lun.yu@rallyhealth.com',
      license="MIT",
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/rallyhealth/pytalite/',
      install_requires=required,
      keywords='pytalite',
      packages=find_packages(include=['pytalite', 'pytalite.*', 'pytalite_spark', 'pytalite_spark.*']),
      )
