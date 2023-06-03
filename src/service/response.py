class Response():

    success = bool()
    message = str()
    code = int()
    data = None

    def __init__(self,
                 success: bool = True,
                 message: str = "Success",
                 code: int = 200,
                 data=None
                 ):
        self.success = success
        self.message = message
        self.code = code
        self.data = data

        if self.data == None:
            self.sendSuccess()
        elif self.data is not None:
            self.sendSuccessWithData()
        elif not self.success:
            self.sendError()

    def sendSuccess(self):
        return {
            "success": self.success,
            "message": self.message,
            "code": self.code
        }

    def sendSuccessWithData(self):
        return {
            "success": self.success,
            "message": self.message,
            "code": self.code,
            "data": self.data
        }

    def sendError(self):
        return {
            "success": self.success,
            "message": self.message,
            "code": self.code,
            "data": self.data
        }
