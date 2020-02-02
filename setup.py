import sys
from pathlib import Path

from setuptools import setup

base_dir = Path(__file__).absolute().parent
sys.path.insert(0, str(base_dir / 'pyFiber3D'))

import config  # import proMAD config without triggering the module __init__.py

read_me = base_dir / 'README.md'
long_description = read_me.read_text(encoding='utf-8')
version = config.version

setup(name=config.app_name,
      version=version,
      description='automated molecular animator',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url=config.url,
      download_url=f'https://github.com/theia-dev/molani/archive/v{version}.zip',
      author=config.app_author,
      author_email='',
      license='MIT',
      entry_points={'console_scripts': ['molani=molani.run:main'], },
      packages=['molani'],
      include_package_data=True,
      install_requires=['arrow', 'numpy', 'matplotlib', 'pillow', 'mpi4py'],
      zip_safe=True,
      keywords=['Rendering', 'Animation', 'Molecular Dynamics'],
      python_requires='~=3.6',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',

          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Multimedia :: Graphics :: 3D Rendering',

          'License :: OSI Approved :: MIT License',

          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3 :: Only'
      ],
      )
