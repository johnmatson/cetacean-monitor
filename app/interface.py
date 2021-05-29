'''
Specifies the Orca Alert GUI. Built using the PyQt5 framework.
'''


import sys
import time
import queue
import threading

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QObject
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QDialog
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
    '''
    Worker thread for AlertUI. Checks status of update_event every
    200 ms to emit a signal if new data is ready to be displayed.
    '''

    update = pyqtSignal()

    def __init__(self, update_event):
        super().__init__()
        self.update_event = update_event

    def run(self):
        while(True):
            if self.update_event.is_set():
                self.update_event.clear()
                self.update.emit()
                
            time.sleep(0.2)


class AlertUI(QMainWindow):
    '''
    AlertUI main window. Displays raw output of CNN model, filtered orca
    output in the form of a 0-100% risk indicator, connect and
    disconnect option buttons, and app status in the status bar.
    '''

    def __init__(self, update_event, predict_pipe, risk_pipe):
        super().__init__()
        self.setWindowTitle('Whale Alert')
        self.setFixedSize(400,600)
        self.general_layout = QVBoxLayout()
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.general_layout)

        self.create_header()
        self.create_model_output()
        self.create_risk_indicator()
        self.create_buttons()
        self.create_status_bar()

        self.connect_button.clicked.connect(self.connect_menu)

        # self.update_status_bar('status2')

        self.predict_pipe = predict_pipe
        self.risk_pipe = risk_pipe

        self.thread = QThread()
        self.worker = AlertWorker(update_event)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.update.connect(self.update_display)
        self.thread.start()

        self.show()

    def create_header(self):
        self.header = QLabel('<h1>Orca Detection</h1>')
        self.general_layout.addWidget(self.header, alignment=Qt.AlignCenter)

    def create_model_output(self):
        self.model_label = QLabel('<h2>Neural network output</h2>')
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

        self.bar = QRoundProgressBar()
        self.bar.setFixedSize(300, 300)

        self.bar.setDataPenWidth(3)
        self.bar.setOutlinePenWidth(3)
        self.bar.setDonutThicknessRatio(0.85)
        self.bar.setDecimals(1)
        self.bar.setFormat('Risk:\n  %p %  ')
        # self.bar.resetFormat()
        self.bar.setNullPosition(270)
        self.bar.setBarStyle(QRoundProgressBar.StyleDonut)
        self.bar.setDataColors([(0., QtGui.QColor.fromRgb(0,255,0)), (0.2, QtGui.QColor.fromRgb(255,255,0)), (0.7, QtGui.QColor.fromRgb(255,0,0))])

        self.bar.setRange(0, 100)
        self.bar.setValue(0)

        self.general_layout.addWidget(self.bar, alignment=Qt.AlignCenter)

    def create_status_bar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)

    def update_status_bar(self, message):
        self.status.showMessage(message)

    def create_buttons(self):
        self.buttons_layout = QHBoxLayout()
        self.connect_button = QPushButton('Connect')
        self.disconnect_button = QPushButton('Disconnect')
        self.connect_button.setMaximumSize(self.disconnect_button.minimumSizeHint())
        self.disconnect_button.setMaximumSize(self.disconnect_button.minimumSizeHint())
        # self.disconnect_button.setEnabled(False)
        # self.connect_button.setDefault(True)
        self.buttons_layout.addWidget(self.disconnect_button)
        self.buttons_layout.addWidget(self.connect_button)
        self.general_layout.addLayout(self.buttons_layout)

    def update_display(self):
        prediction = self.predict_pipe.get()
        self.orca.setText('{:2.1%}'.format(prediction[0][0]))
        self.silence.setText('{:2.1%}'.format(prediction[0][1]))
        self.speech.setText('{:2.1%}'.format(prediction[0][2]))
        self.water.setText('{:2.1%}'.format(prediction[0][3]))

        risk = self.risk_pipe.get()
        self.bar.setValue(100*risk)

    def connect_menu(self):
        self.dialog = ConnectUI()
        self.dialog.show()


class ConnectUI(QDialog):
    '''
    Dialog element used to specify the operating mode of AlertUI during
    connection setup.
    '''

    def __init__(self, parent=AlertUI):
        super().__init__()
        self.setWindowTitle('Connection Setup')
        self.general_layout = QVBoxLayout()

        self.stream = QPushButton('Stream')
        self.mic = QPushButton('Mic')
        self.disk = QPushButton('Disk')
        self.disk.setDefault(True)

        self.stream.clicked.connect()

        self.general_layout.addWidget(self.stream)
        self.general_layout.addWidget(self.mic)
        self.general_layout.addWidget(self.disk)

        self.setLayout(self.general_layout)


class QRoundProgressBar(QWidget):
    '''
    A circular PyQt progress bar, based heavily on code from Alexander
    Lutsenko, available here:
    https://stackoverflow.com/questions/33577068/any-pyqt-circular-progress-bar
    which is a ported adaption of QRoundProgressBar for Qt, created by
    Sintegrial Technologies and available here:
    https://sourceforge.net/projects/qroundprogressbar/
    '''

    StyleDonut = 1
    StylePie = 2
    StyleLine = 3

    PositionLeft = 180
    PositionTop = 90
    PositionRight = 0
    PositionBottom = -90

    UF_VALUE = 1
    UF_PERCENT = 2
    UF_MAX = 4

    def __init__(self):
        super().__init__()
        self.min = 0.
        self.max = 100.
        self.value = 25.

        self.nullPosition = self.PositionTop
        self.barStyle = self.StyleDonut
        self.outlinePenWidth =1
        self.dataPenWidth = 1
        self.rebuildBrush = False
        self.format = "%p%"
        self.decimals = 1
        self.updateFlags = self.UF_PERCENT
        self.gradientData = []
        self.donutThicknessRatio = 0.75

    def setRange(self, min, max):
        self.min = min
        self.max = max

        if self.max < self.min:
            self.max, self.min = self.min, self.max

        if self.value < self.min:
            self.value = self.min
        elif self.value > self.max:
            self.value = self.max

        if not self.gradientData:
            self.rebuildBrush = True
        self.update()

    def setMinimun(self, min):
        self.setRange(min, self.max)

    def setMaximun(self, max):
        self.setRange(self.min, max)

    def setValue(self, val):
        if self.value != val:
            if val < self.min:
                self.value = self.min
            elif val > self.max:
                self.value = self.max
            else:
                self.value = val
            self.update()

    def setNullPosition(self, position):
        if position != self.nullPosition:
            self.nullPosition = position
            if not self.gradientData:
                self.rebuildBrush = True
            self.update()

    def setBarStyle(self, style):
        if style != self.barStyle:
            self.barStyle = style
            self.update()

    def setOutlinePenWidth(self, penWidth):
        if penWidth != self.outlinePenWidth:
            self.outlinePenWidth = penWidth
            self.update()

    def setDataPenWidth(self, penWidth):
        if penWidth != self.dataPenWidth:
            self.dataPenWidth = penWidth
            self.update()

    def setDataColors(self, stopPoints):
        if stopPoints != self.gradientData:
            self.gradientData = stopPoints
            self.rebuildBrush = True
            self.update()

    def setFormat(self, format):
        if format != self.format:
            self.format = format
            self.valueFormatChanged()

    def resetFormat(self):
        self.format = ''
        self.valueFormatChanged()

    def setDecimals(self, count):
        if count >= 0 and count != self.decimals:
            self.decimals = count
            self.valueFormatChanged()

    def setDonutThicknessRatio(self, val):
        self.donutThicknessRatio = max(0., min(val, 1.))
        self.update()

    def paintEvent(self, event):
        outerRadius = min(self.width(), self.height())
        baseRect = QtCore.QRectF(1, 1, outerRadius-2, outerRadius-2)

        buffer = QtGui.QImage(outerRadius, outerRadius, QtGui.QImage.Format_ARGB32)
        buffer.fill(0)

        p = QtGui.QPainter(buffer)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        # data brush
        self.rebuildDataBrushIfNeeded()

        # background
        # self.drawBackground(p, buffer.rect())

        # base circle
        self.drawBase(p, baseRect)

        # data circle
        arcStep = 360.0 / (self.max - self.min) * self.value
        self.drawValue(p, baseRect, self.value, arcStep)

        # center circle
        innerRect, innerRadius = self.calculateInnerRect(baseRect, outerRadius)
        self.drawInnerBackground(p, innerRect)

        # text
        self.drawText(p, innerRect, innerRadius, self.value)

        # finally draw the bar
        p.end()

        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, buffer)

    def drawBackground(self, p, baseRect):
        p.fillRect(baseRect, self.palette().background())

    def drawBase(self, p, baseRect):
        bs = self.barStyle
        if bs == self.StyleDonut:
            p.setPen(QtGui.QPen(self.palette().shadow().color(), self.outlinePenWidth))
            p.setBrush(self.palette().base())
            p.drawEllipse(baseRect)
        elif bs == self.StylePie:
            p.setPen(QtGui.QPen(self.palette().base().color(), self.outlinePenWidth))
            p.setBrush(self.palette().base())
            p.drawEllipse(baseRect)
        elif bs == self.StyleLine:
            p.setPen(QtGui.QPen(self.palette().base().color(), self.outlinePenWidth))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(baseRect.adjusted(self.outlinePenWidth/2, self.outlinePenWidth/2, -self.outlinePenWidth/2, -self.outlinePenWidth/2))

    def drawValue(self, p, baseRect, value, arcLength):
        # nothing to draw
        if value == self.min:
            return

        # for Line style
        if self.barStyle == self.StyleLine:
            p.setPen(QtGui.QPen(self.palette().highlight().color(), self.dataPenWidth))
            p.setBrush(Qt.NoBrush)
            p.drawArc(baseRect.adjusted(self.outlinePenWidth/2, self.outlinePenWidth/2, -self.outlinePenWidth/2, -self.outlinePenWidth/2),
                      self.nullPosition * 16,
                      -arcLength * 16)
            return

        # for Pie and Donut styles
        dataPath = QtGui.QPainterPath()
        dataPath.setFillRule(Qt.WindingFill)

        # pie segment outer
        dataPath.moveTo(baseRect.center())
        dataPath.arcTo(baseRect, self.nullPosition, -arcLength)
        dataPath.lineTo(baseRect.center())

        p.setBrush(self.palette().highlight())
        p.setPen(QtGui.QPen(self.palette().shadow().color(), self.dataPenWidth))
        p.drawPath(dataPath)

    def calculateInnerRect(self, baseRect, outerRadius):
        # for Line style
        if self.barStyle == self.StyleLine:
            innerRadius = outerRadius - self.outlinePenWidth
        else:    # for Pie and Donut styles
            innerRadius = outerRadius * self.donutThicknessRatio

        delta = (outerRadius - innerRadius) / 2.
        innerRect = QtCore.QRectF(delta, delta, innerRadius, innerRadius)
        return innerRect, innerRadius

    def drawInnerBackground(self, p, innerRect):
        if self.barStyle == self.StyleDonut:
            p.setBrush(self.palette().alternateBase())

            cmod = p.compositionMode()
            p.setCompositionMode(QtGui.QPainter.CompositionMode_Source)

            p.drawEllipse(innerRect)

            p.setCompositionMode(cmod)

    def drawText(self, p, innerRect, innerRadius, value):
        if not self.format:
            return

        text = self.valueToText(value)

        # !!! to revise
        f = self.font()
        # f.setPixelSize(innerRadius * max(0.05, (0.35 - self.decimals * 0.08)))
        f.setPixelSize(innerRadius * 1.8 / len(text))
        p.setFont(f)

        textRect = innerRect
        p.setPen(self.palette().text().color())
        p.drawText(textRect, Qt.AlignCenter, text)

    def valueToText(self, value):
        textToDraw = self.format

        format_string = '{' + ':.{}f'.format(self.decimals) + '}'

        if self.updateFlags & self.UF_VALUE:
            textToDraw = textToDraw.replace("%v", format_string.format(value))

        if self.updateFlags & self.UF_PERCENT:
            percent = (value - self.min) / (self.max - self.min) * 100.0
            textToDraw = textToDraw.replace("%p", format_string.format(percent))

        if self.updateFlags & self.UF_MAX:
            m = self.max - self.min + 1
            textToDraw = textToDraw.replace("%m", format_string.format(m))

        return textToDraw

    def valueFormatChanged(self):
        self.updateFlags = 0;

        if "%v" in self.format:
            self.updateFlags |= self.UF_VALUE

        if "%p" in self.format:
            self.updateFlags |= self.UF_PERCENT

        if "%m" in self.format:
            self.updateFlags |= self.UF_MAX

        self.update()

    def rebuildDataBrushIfNeeded(self):
        if self.rebuildBrush:
            self.rebuildBrush = False

            dataBrush = QtGui.QConicalGradient()
            dataBrush.setCenter(0.5,0.5)
            dataBrush.setCoordinateMode(QtGui.QGradient.StretchToDeviceMode)

            for pos, color in self.gradientData:
                dataBrush.setColorAt(1.0 - pos, color)

            # angle
            dataBrush.setAngle(self.nullPosition)

            p = self.palette()
            p.setBrush(QtGui.QPalette.Highlight, dataBrush)
            self.setPalette(p)


if __name__ == '__main__':
    update_event = threading.Event()
    predict_pipe = queue.Queue(maxsize=10)
    risk_pipe = queue.Queue(maxsize=10)

    app = QApplication(sys.argv)
    win = AlertUI(update_event, predict_pipe, risk_pipe)
    sys.exit(app.exec())
    