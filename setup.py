from setuptools import setup

with open('requirements.txt') as fh:
    REQUIREMENTS = fh.read().split('\n')

setup(name='dwaveip',
      version='0.1.2',
      description='A package to handle integers with Dwave',
      author='Hayk Sargsyan',
      # author_email='',
      url='https://github.com/hay-k/dwave-ip',
      packages=['dwaveip'],
      license='MIT License',
      install_requires=REQUIREMENTS,
      python_requires='>=3.6'
      )