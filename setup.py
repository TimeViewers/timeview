from setuptools import setup

requirements = ["numpy", #""intel-numpy",
                "scipy",
                "sqlalchemy",
                "numba",
                "qtpy",
                "pyqtgraph",
                "qtawesome",
                "pyedflib"]

test_requirements = ["pytest",
                     "pytest-qt",
                     "pytest-runner"]


setup(
    # meta-data
    name='TimeView',
    version='0.1.0',
    description="A GUI application to view and analyze time series signal data",
    author=["Alexander Kain", "Ognyan Moore"],
    author_email=['lxkain@gmail.com', 'ognyan.moore@gmail.com'],
    url='https://github.com/lxlain/timeview',
    keywords='timeview gui pyqt signal spectrogram',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Utilities',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License'
    ],
    license='MIT',
    # app contents
    packages=['timeview',
              'timeview.dsp',
              'timeview.gui',
              'timeview.manager'],
    include_package_data=True,
    # launching
    entry_points={
        'gui_scripts': [
            'timeview = timeview.__main__'
        ]
    },
    # dependencies
    install_requires=requirements,
    tests_require=test_requirements,
    extras_require={
        'dev': [
            "numpydoc",
            "flake8",
            "mypy",
            "pylint"
        ],
        'test': test_requirements
    },
    python_requires=">=3.6.0",
    # setup_requires=["pytest-runner"],
)
