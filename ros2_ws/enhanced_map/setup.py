from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'enhanced_map'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'matplotlib',
        'h5py',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='javideus',
    maintainer_email='javi.rm2005@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'map_publisher = enhanced_map.map_publisher:main',
        ],
    },
)
