# evidence.py - the real POST /evidence: what does the picture FILE say about itself?
#
# Only EXIF is built so far. EXIF is the small block of text a camera or phone writes into a photo
# (camera make, model, date, ...). The other three parts of the answer are not built yet:
#   C2PA (signed "content credentials")  -> status "not_checked"
#   ELA and FFT pictures                 -> null
# What this file does:
#   1. open the upload (PNG or JPG; common.read_upload already checked the type and the 5 MB limit);
#      a picture that cannot be read, or has more than 40 million pixels, gives 400
#   2. read the EXIF tags with Pillow and turn each one into a short piece of text
#   3. PRIVACY: GPS coordinates are NEVER returned. If the file has GPS data we only say it is there.
# Nothing is stored. EXIF is easy to remove or to fake, so it is evidence, never proof.

import io

from PIL import Image
from PIL.ExifTags import TAGS

from api.common import ApiError
from api.compare import MAX_PIXELS          # the same "too many pixels" limit as /compare and /detect

MAX_FIELDS = 30                  # at most this many tags are returned
MAX_VALUE_LENGTH = 100           # a longer value is cut after this many characters
EXIF_IFD = 0x8769                # the tag that points to the second block of camera settings (ExposureTime, ...)
GPS_IFD = 0x8825                 # the tag that points to the GPS block. We never read inside it.
SKIPPED = {EXIF_IFD, GPS_IFD, 0xA005}      # 0xA005 = another pointer. Pointers are not real values.

EXIF_NOTE = "Missing EXIF is weak evidence; present EXIF does not mean real."
C2PA_NOTE = "A missing manifest means nothing."
NOTE = "Evidence, not proof. Nothing is stored."


def open_picture(data):
    """The uploaded bytes -> a Pillow picture (EXIF still attached). Raises ApiError(400) if bad or too big."""
    try:
        picture = Image.open(io.BytesIO(data))              # this only reads the file header
        too_big = picture.width * picture.height > MAX_PIXELS
        if not too_big:
            picture.load()                                  # this reads all the pixels (a cut-off file fails here)
    except Exception:
        raise ApiError(400, "the uploaded image could not be read as a picture")
    if too_big:
        raise ApiError(400, f"the uploaded image has more than {MAX_PIXELS // 1_000_000} million pixels")
    return picture


def short_text(value):
    """One EXIF value -> a short readable text, or None if it should be skipped (binary data, empty)."""
    if isinstance(value, bytes):
        return None                                         # binary blobs (MakerNote and similar) are skipped
    text = "".join(ch for ch in str(value) if ch.isprintable()).strip()
    return text[:MAX_VALUE_LENGTH] if text else None


def read_fields(picture):
    """{tag name: short text} for the readable EXIF tags, plus one 'GPSInfo' line if the file has GPS data."""
    try:
        exif = picture.getexif()
        has_gps = GPS_IFD in exif
        tags = dict(exif)
        tags.update(exif.get_ifd(EXIF_IFD))                 # the second block: DateTimeOriginal, ExposureTime, ...
    except Exception:
        return {}                                           # a broken EXIF block: we say "nothing readable"
    room = MAX_FIELDS - 1 if has_gps else MAX_FIELDS        # keep one place free for the GPS line
    fields = {}
    for number, value in tags.items():
        name, text = TAGS.get(number), short_text(value)
        if name and number not in SKIPPED and text and len(fields) < room:
            fields[name] = text
    if has_gps:
        fields["GPSInfo"] = "present (coordinates hidden)"  # never the numbers
    return fields


def evidence(data):
    """Build the whole /evidence answer (see api/contract.json). The upload is already checked."""
    fields = read_fields(open_picture(data))
    return {
        "exif": {"found": len(fields) > 0, "fields": fields, "does_not_prove": EXIF_NOTE},
        "c2pa": {"status": "not_checked", "summary": None, "does_not_prove": C2PA_NOTE},
        "ela_png_base64": None,
        "fft_png_base64": None,
        "note": NOTE,
    }
