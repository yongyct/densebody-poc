class ModelNotFoundError(Exception):
    '''
    Custom exception raised when model requested is not found
    '''
    def __init__(self, message):
        super().__init__(message)
        self.message = message
