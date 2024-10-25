from setuptools import find_packages, setup

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
        "console_scripts": [
            "perception = ros2.threads.perception:main",
            "correlator = ros2.threads.correlator:main",
            "navigation = ros2.threads.navigation:main",
            "recorder   = ros2.threads.recorder:main",
            "node       = ros2.node:main",
        ],
    },
)
