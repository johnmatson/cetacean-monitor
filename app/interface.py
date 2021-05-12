import sys
import queue
import time

from PyQt5.QtCore import QObject, Qt
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QStatusBar
from PyQt5.QtWidgets import QLabel


class AlertWorker(QObject):

    update = pyqtSignal()

    def __init__(self, update_event):
        super().__init__()
        self.update_event = update_event

    def run(self):
        while(True):
            if self.update_event.is_set():
                self.update_event.clear()
                self.update.emit()

class AlertUI(QMainWindow):

    def __init__(self, update_event, predict_pipe, alert_pipe):
        super().__init__()
        self.setWindowTitle('Whale Alert')
        self.general_layout = QVBoxLayout()
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.general_layout)

        self.create_header()
        self.create_model_output()
        self.create_risk_indicator()
        self.create_buttons()
        self.create_status_bar()

        # self.update_status_bar('status2')

        self.predict_pipe = predict_pipe
        self.alert_pipe = alert_pipe

        self.thread = QThread()
        self.worker = AlertWorker(update_event)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.update.connect(self.update_display)
        self.thread.start()

        self.show()

    def create_header(self):
        self.header = QLabel('<h1>Orca Detection</h1>')
        self.header.setAlignment(Qt.AlignCenter)
        self.general_layout.addWidget(self.header)

    def create_model_output(self):
        self.model_label = QLabel('<h2>Neurel network output</h2>')
        self.general_layout.addWidget(self.model_label)

        self.model_layout = QGridLayout()

        self.model_layout.addWidget(QLabel('<h3>Orca</h3>'),0,0)
        self.model_layout.addWidget(QLabel('<h3>Silence</h3>'),0,1)
        self.model_layout.addWidget(QLabel('<h3>Speech</h3>'),0,2)
        self.model_layout.addWidget(QLabel('<h3>Water</h3>'),0,3)

        self.orca = QLabel('0%')
        self.silence = QLabel('0%')
        self.speech = QLabel('0%')
        self.water = QLabel('0%')

        self.model_layout.addWidget(self.orca,1,0)
        self.model_layout.addWidget(self.silence,1,1)
        self.model_layout.addWidget(self.speech,1,2)
        self.model_layout.addWidget(self.water,1,3)

        self.general_layout.addLayout(self.model_layout)

    def create_risk_indicator(self):
        self.risk_label = QLabel('<h2>Risk indicator</h2>')
        self.general_layout.addWidget(self.risk_label)

    def create_status_bar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)

    def update_status_bar(self, message):
        self.status.showMessage(message)

    def create_buttons(self):
        self.buttons_layout = QHBoxLayout()
        self.connect = QPushButton('Connect')
        self.disconnect = QPushButton('Disconnect')
        self.buttons_layout.addWidget(self.disconnect)
        self.buttons_layout.addWidget(self.connect)
        self.general_layout.addLayout(self.buttons_layout)

    def update_display(self):
        prediction = self.predict_pipe.get()
        self.orca.setText('{:2.1%}'.format(prediction[0][0]))
        self.silence.setText('{:2.1%}'.format(prediction[0][1]))
        self.speech.setText('{:2.1%}'.format(prediction[0][2]))
        self.water.setText('{:2.1%}'.format(prediction[0][3]))

        alert = self.alert_pipe.get()
        # print(self.predict_pipe.qsize())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = AlertUI()
    sys.exit(app.exec())
    