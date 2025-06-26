class BaseAPIException(Exception):
    """Base exception for API errors"""
    
    def __init__(self, status_code: int = 500, detail: str = "Internal server error"):
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.detail)


class NotFoundException(BaseAPIException):
    """Exception for resource not found errors"""
    
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(status_code=404, detail=detail)


class BadRequestException(BaseAPIException):
    """Exception for bad request errors"""
    
    def __init__(self, detail: str = "Bad request"):
        super().__init__(status_code=400, detail=detail)


class ValidationException(BaseAPIException):
    """Exception for validation errors"""
    
    def __init__(self, detail: str = "Validation error"):
        super().__init__(status_code=422, detail=detail)


class ServiceException(BaseAPIException):
    """Exception for service errors"""
    
    def __init__(self, detail: str = "Service error"):
        super().__init__(status_code=500, detail=detail)
