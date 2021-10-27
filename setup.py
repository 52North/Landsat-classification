# =================================================================
# Copyright (C) 2021-2021 52°North Spatial Information Research GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# https://docs.python.org/3/distutils/setupscript.html
#
# =================================================================

from setuptools import setup, find_packages


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


# loading requirements
requirements = list(parse_requirements('requirements.txt'))

setup(
    name="landsatpredictor",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    version="0.1.0",
    description="pygeoapi processor plugin for Landsat land cover classification",
    long_description="landsat ML model for land cover classification",
    long_description_content_type="text/markdown",
    license="Apache License, Version 2.0",
    url="https://github.com/52North/Landsat-classification",
    keywords=["Landsat", "pygeoapi", "EO", "Landsat Level 2 Collection 2", "machine learning"],
    install_requires=requirements,
    test_suite="tests",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache License, Version 2.0',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ]
)
