'''
Runs on the Raspberry Pi. Reads audio from external USB
audio interface or audio file on disk. Establishes socket
connection with remote client and streams audio over
connection. Call file from command line with a mode
configuration argument of 'disk' or 'mic' or with no
arguments to run in disk mode by default.
'''


import sys


def transmit(mode='disk', audio_path='app/datasets/full-clips/001A-short.wav'):

    # configures module to source audio from local audio file
    if mode == 'disk':
        # ADD SOCKET CODE HERE
        pass

    # configures module to source audio from local audio input
    elif mode == 'mic':
        pass


if __name__ == "__main__":

    if len(sys.argv) < 2:
        transmit()
    elif len(sys.argv) == 2:
        transmit(mode=sys.argv[1])
