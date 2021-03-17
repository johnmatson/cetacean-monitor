'''
User-defined exceptions for use in cetacean-monitor project.
'''

class Error(Exception):
    '''Base class for other exceptions'''
    pass

class EndOfFileError(Error):
    '''
    Raised when not enough data remains in disk
    audio file to copy to clip variable
    '''
    pass
