class InvalidJsonConfigError(Exception):
    '''
    Custom exception raised when json config passed is invalid
    '''
    def __init__(self, message):
        super().__init__(message)
        self.message = message
