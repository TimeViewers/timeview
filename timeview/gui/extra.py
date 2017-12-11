from qtpy import QtWidgets, QtGui


class Widget(QtWidgets.QWidget):

    def __init__(self, text: str="", panel_info=None):
        super().__init__()
        layout = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel(f"Hello, I'm {text}!")
        layout.addWidget(self.label)
        self.setLayout(layout)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label.setFont(font)
        self.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                           QtWidgets.QSizePolicy.MinimumExpanding)
        self.setStyleSheet("""
        QWidget {
            padding: 0px;
            margin: 0px;
            background: darkGreen;
        }
        """)


class Spacer(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
