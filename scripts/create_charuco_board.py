import cv2
import cv2.aruco as aruco
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
import numpy as np

# Parameters for the ChArUco board
SQUARE_SIZE_MM = 6  # Desired square size in mm
MARKER_SIZE_MM = 4  # Desired marker size in mm
SQUARES_X = 8  # Number of squares along x-axis
SQUARES_Y = 8  # Number of squares along y-axis
MARGIN_MM = 10  # Margin around the board

# Create the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Define the board size as a tuple (number of squares in x and y directions)
board_size = (SQUARES_X, SQUARES_Y)

# Create the Charuco board using the updated constructor
charuco_board = aruco.CharucoBoard(
    size=board_size,
    squareLength=SQUARE_SIZE_MM / 1000.0,  # in meters
    markerLength=MARKER_SIZE_MM / 1000.0,  # in meters
    dictionary=aruco_dict,
)

# Generate the board image using OpenCV
img_size = (
    SQUARES_X * 300,
    SQUARES_Y * 300,
)  # Higher resolution for print quality
img = charuco_board.generateImage(img_size)

# Save the board image as a PNG for reference
cv2.imwrite("charuco_board.png", img)

# Calculate the final PDF size including margins
pdf_width = (SQUARES_X * SQUARE_SIZE_MM) + (2 * MARGIN_MM)  # mm
pdf_height = (SQUARES_Y * SQUARE_SIZE_MM) + (2 * MARGIN_MM)  # mm

# Create a PDF using ReportLab with precise dimensions
pdf_path = "charuco_board_accurate.pdf"
c = canvas.Canvas(pdf_path, pagesize=(pdf_width * mm, pdf_height * mm))

# Convert the OpenCV image to a format suitable for PDF
_, buffer = cv2.imencode(".png", img)
image_data = buffer.tobytes()

# Calculate the exact position for centering the image
x_offset = MARGIN_MM * mm
y_offset = MARGIN_MM * mm
image_width = (SQUARES_X * SQUARE_SIZE_MM) * mm
image_height = (SQUARES_Y * SQUARE_SIZE_MM) * mm

# Draw the ChArUco board image onto the PDF
c.drawImage("charuco_board.png", x_offset, y_offset, image_width, image_height)

# Save the PDF
c.showPage()
c.save()
print(f"Saved accurate PDF: {pdf_path}")
