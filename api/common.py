# common.py - two small things that more than one API file needs:
#   1. ApiError: how our code says "answer with this HTTP status and this message".
#   2. read_upload: reads an uploaded image and checks it (PNG or JPG, at most 5 MB).
# They live here so that main.py, generators.py and stubs.py do not import each other in a circle.

# The biggest image a user may upload (proposed, to tune).
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

# The first bytes of a real PNG and a real JPG file. We check these, not the file name.
PNG_START = b"\x89PNG\r\n\x1a\n"
JPG_START = b"\xff\xd8\xff"


class ApiError(Exception):
    """Raise this to answer with a status code (400, 404, 503, ...) and a plain-English message."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


async def read_upload(file):
    """Read one uploaded file and return its bytes. Raises ApiError(400) if it is not OK."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)     # read one byte more than the limit
    if len(data) == 0:
        raise ApiError(400, "the uploaded image is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ApiError(400, f"the uploaded image is bigger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not (data.startswith(PNG_START) or data.startswith(JPG_START)):
        raise ApiError(400, "the uploaded image must be a PNG or JPG file")
    return data
