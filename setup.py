from setuptools import setup

setup(name='motifs',
      version='0.1',
      description='Finding temporal motifs in data',
      long_description='Finding temporal motifs in data',
      classifiers=[
        'Programming Language :: Python :: 3.5',
      ],
      keywords='funniest joke comedy flying circus',
      url='tbc',
      author='Andrew Mellor',
      author_email='mellor91@hotmail.co.uk',
      license='MIT',
      packages=['motifs'],
      install_requires=[
          'networkx',
          'pandas',
          'numpy'
      ],
      include_package_data=True,
      zip_safe=False)