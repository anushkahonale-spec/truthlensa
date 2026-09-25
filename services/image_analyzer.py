from PIL import Image
from io import BytesIO
import cv2
import numpy as np


def analyze_image(image_bytes):
    """
    TruthLens image analysis and basic forensic signal detection.
    """

    try:
        # -----------------------------------
        # 1. Read image with Pillow
        # -----------------------------------

        image = Image.open(BytesIO(image_bytes))

        width, height = image.size
        image_format = image.format
        mode = image.mode

        file_size_kb = len(image_bytes) / 1024

        # -----------------------------------
        # 2. Read image with OpenCV
        # -----------------------------------

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        cv_image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if cv_image is None:
            return {
                "status": "error",
                "message": "Unable to decode image."
            }

        # -----------------------------------
        # 3. Convert to grayscale
        # -----------------------------------

        gray = cv2.cvtColor(
            cv_image,
            cv2.COLOR_BGR2GRAY
        )

        # -----------------------------------
        # 4. Calculate noise level
        # -----------------------------------

        noise = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        # -----------------------------------
        # 5. Calculate edge density
        # -----------------------------------

        edges = cv2.Canny(
            gray,
            100,
            200
        )

        edge_pixels = np.count_nonzero(edges)

        total_pixels = edges.shape[0] * edges.shape[1]

        edge_density = edge_pixels / total_pixels

        # -----------------------------------
        # 6. Check EXIF metadata
        # -----------------------------------

        exif_data = image.getexif()

        if exif_data:
            metadata_status = "EXIF metadata present"
        else:
            metadata_status = "No EXIF metadata found"

        # -----------------------------------
        # 7. Generate forensic signals
        # -----------------------------------

        signals = []

        if width < 200 or height < 200:
            signals.append(
                "Low-resolution image"
            )

        if file_size_kb > 10000:
            signals.append(
                "Very large image file"
            )

        if noise < 10:
            signals.append(
                "Very low high-frequency variation"
            )

        if edge_density < 0.01:
            signals.append(
                "Very low edge density"
            )

        # -----------------------------------
        # 8. Determine result
        # -----------------------------------

        # These signals alone cannot prove
        # that an image is fake.

        if len(signals) >= 2:

            status = "uncertain"

            message = (
                "The image contains multiple forensic signals "
                "that require further verification."
            )

        else:

            status = "uncertain"

            message = (
                "No strong basic manipulation signals detected. "
                "This does not prove the image is genuine."
            )

        # -----------------------------------
        # 9. Return complete result
        # -----------------------------------

        return {

            "status": status,

            "message": message,

            "image": {

                "width": width,

                "height": height,

                "format": image_format,

                "mode": mode,

                "file_size_kb": round(
                    file_size_kb,
                    2
                ),

                "metadata": metadata_status

            },

            "forensics": {

                "noise_level": round(
                    float(noise),
                    2
                ),

                "edge_density": round(
                    float(edge_density),
                    4
                )

            },

            "signals": signals

        }

    except Exception as e:

        return {

            "status": "error",

            "message": (
                f"Unable to analyze image: {str(e)}"
            )

        }