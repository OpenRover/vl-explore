from setuptools import find_packages, setup
from warnings import filterwarnings

filterwarnings("ignore", category=DeprecationWarning)

package_name = "perception"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["ros2/resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yuxuan",
    maintainer_email="robotics@z-yx.cc",
    description="TODO: Package description",
    license="MIT",
    entry_points={
        "console_scripts": ["ros2 = ros2.main:main"],
    },
)
