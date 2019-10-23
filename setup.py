from setuptools import setup 

setup(name='TensorSAT',
      install_requires=['numpy','tensorflow','randSAT'],
      dependency_links=['https://github.com/mister-bailey/RandSAT/tarball/master#egg=randSAT']
      )