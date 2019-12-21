
"""
The build/compilations setup
>> pip install -r requirements.txt
>> python setup.py install
"""

import pip
import logging
import pkg_resources
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

    # you may need setuptools instead of distutils

def _parse_requirements(file_path):
    pip_ver = pkg_resources.get_distribution('pip').version
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path,
                                         session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]

# parse_requirements() returns generator of pip.req.InstallRequirement objects
try:
    install_reqs = _parse_requirements("requirements.txt")
except Exception:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = []


setup(
    name='davinci',
    version='0.1',
    #url='https://github.com/matterport/Mask_RCNN',
    author='Yubin Xie',
    author_email='yoobintse@gmail.com',
    license='MIT',
    description='Mask R-CNN for object detection and instance segmentation',
    packages=["davinci"],
    install_requires=install_reqs,
    include_package_data=True,
    python_requires='>=3.6',
    long_description="""For Vectra analysis""",
#    scripts = [
 #       'scripts/myscript.sh'
#    ]
)