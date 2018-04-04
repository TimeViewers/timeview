from setuptools import setup

requirements = ["intel-numpy",
                "scipy",
                "sqlalchemy",
                "numba",
                "pyqt5",
                "qtpy",
                "pyqtgraph",
                "qtawesome",
                "pyedflib"]

setup(
    name='TimeView',
    version='0.1.0',
    description="A GUI application to view and analyze time series signal data",
    author=["Ognyan Moore", "Alex Kain"],
    author_email=['ognyan.moore@gmail.com', 'lxkain@gmail.com'],
    url='https://github.com/lxlain/timeview',
    packages=['timeview',
              'timeview.dsp',
              'timeview.gui',
              'timeview.manager'],
    entry_points={
        'gui_scripts': [
            'timeview = timeview:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    keywords='timeview gui pyqt signal spectrogram',
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"]
)
