from setuptools import setup
# from setuptools import find_packages

setup(name='pathmind',
      version='0.1',
      description='Python Simulations ',
      url='https://github.com/SkymindIO/pathmind_api',
      download_url='https://github.com/pathmindai/pathmind_api/tarball/0.1',
      author='Max Pumperla',
      author_email='max@pathmind.com',
      install_requires=['numpy', 'pyyaml'],
      packages=['pathmind'],  # find_packages()
      license='MIT',
      zip_safe=False,
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3'
    ])
