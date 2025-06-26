# Copyright (c) 2023, zw and contributors
# For license information, please see license.txt

import frappe
from frappe.model.document import Document
import cv2
import numpy as np 
from PIL import Image
import os
from pdf2image import convert_from_path
from frappe.utils import cstr, getdate
import pytesseract
import re
from datetime import datetime
class FileManager(Document):

    def before_save(self):
        path = os.getcwd()
        site = cstr(frappe.local.site)
        path=f'{path}/{site}{self.file}'
        if os.path.isfile(path):
            try:
                file_extension = os.path.splitext(path)[1]
                if file_extension.lower() in {'.pdf'}:
                    doc = convert_from_path(path)
              
                    text = ""
                    for page in doc:
                        text += pytesseract.image_to_string(page)
                    self.scanned_contents = text
                else:
                    img = cv2.imread(path)
                    # gray = self.get_grayscale(img)
                    # thresh = self.thresholding(gray)
                    new = self.preprocess_image(img)
                    custom_config = r'--oem 3 --psm 6'
                    raw = pytesseract.image_to_string(img, config=custom_config)
                    dicts = self.parse_ocr_data(raw)
                    self.scanned_contents = str(dicts)
            except Exception as e:
                frappe.msgprint(f'Error in file. {e}')
    
        

    def parse_ocr_data(self, raw):
        if not raw:
            return

        # Example OCR response:
        # P<PAKALI<<RAHMAH<<<<<<<L<LLLLLLLLLLLLLLLLLL<LK
        # MA18358115PAK2208109F28052324220121655812<58
        # frappe.errprint(f"raw {raw[-100:]}")
        # frappe.errprint(f"raw lines {raw[-100:].splitlines()}")
        # Split the response into lines
        lines = raw[-100:].splitlines()
        if len(lines) < 2:
            frappe.msgprint("Invalid OCR response format.")
            return

        line1 = lines[1]  # P<PAKALI<<RAHMAH<<<<<<<L<LLLLLLLLLLLLLLLLLL<LK
        line2 = lines[2]  # MA18358115PAK2208109F28052324220121655812<58
        # frappe.errprint(f"len line1 {len(line1)}")
        # frappe.errprint(f"len line2 {len(line2)}")
        if len(line1) < 15 or len(line2) <15 :
            line1 = lines[2]
            line2 = lines[3]


        # frappe.errprint(f"line1 {line1}")
        # frappe.errprint(f"line2 {line2}")
        # Extract fields
        try:
            # Passport Type (P)
            passport_type = line1[0]

            # Last Name (PAKALI)
            # Nationality (PAK)
            nationality = line2[10:13]
            
            last_name = line1.split("<<")[0][1:].split(nationality)[-1]  # Remove 'P<' and take until '<<'

            # First Name (RAHMAH)
            first_name = line1.split("<<")[1].split("<")[0]

            # Passport Number (MA18358115)
            passport_number = line2[:9]

            # Date of Birth (220810 -> 10-08-2022)
            dob_str = line2[13:19]  # YYMMDD format
            try:
                dob = datetime.strptime(dob_str, "%y%m%d").strftime("%d-%m-%Y")
            except:
                dob = ''

            # Sex (F)
            sex = line2[20]

            # Date of Expiry (280523 -> 23-05-2028)
            expiry_str = line2[21:27]  # YYMMDD format
            try:
                expiry = datetime.strptime(expiry_str, "%y%m%d").strftime("%d-%m-%Y")
            except:
                expiry = ''
            # CNIC Number (4220121655812)
            cnic = line2[28:41]

            # Create a dictionary with the extracted data
            parsed_data = {
                "passport_type": passport_type,
                "last_name": last_name,
                "first_name": first_name,
                "passport_number": passport_number,
                "nationality": nationality,
                "date_of_birth": dob,
                "sex": sex,
                "date_of_expiry": expiry,
                "cnic": cnic
            }
            return parsed_data

        except Exception as e:
            frappe.msgprint(f"Error parsing OCR data: {e}. Upload new clearer image.")
            
            
    def get_grayscale(self,image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    def remove_noise(self, image):
        return cv2.medianBlur(image, 5)

    # thresholding
    def thresholding(self, image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # dilation
    def dilate(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    # erosion
    def erode(self,image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    # opening - erosion followed by dilation
    def opening(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # canny edge detection
    def canny(self, image):
        return cv2.Canny(image, 100, 200)

    # skew correction
    def deskew(self, image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated

    # template matching
    def match_template(self, image, template):
        return cv2.matchTemplate(image, template,cv2.TM_CCOEFF_NORMED)
    
    def sharpen_image(self, image):
        """
        Sharpens the image using a kernel to enhance edges.
        """
        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def convert_to_black_and_white(self, image):
        """
        Converts the image to black and white using adaptive thresholding.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw_image

    def reduce_blur(self, image):
        """
        Reduces blur using Gaussian blur followed by unsharp masking.
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        
        # Unsharp masking
        unsharp_image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        return unsharp_image

    def preprocess_image(self, image):
        sharpened = self.sharpen_image(image)
        # bw_image = self.convert_to_black_and_white(sharpened)
        final_image = self.reduce_blur(sharpened)
        return final_image
    
@frappe.whitelist()
def scan_passport(file_url):
    path = os.getcwd()
    site = cstr(frappe.local.site)
    path=f'{path}/{site}{file_url}'
    if os.path.isfile(path):
        try:
            file_extension = os.path.splitext(path)[1]
            if file_extension.lower() in {'.pdf'}:
                doc = convert_from_path(path)
              
                text = ""
                for page in doc:
                    text += pytesseract.image_to_string(page)
                return text
            else:
                img = cv2.imread(path)
                # new = preprocess_image(img)
                custom_config = r'--oem 3 --psm 6'
                raw = pytesseract.image_to_string(img, config=custom_config)
                parsed_data = parse_ocr_data(raw)
                if not parsed_data:
                    frappe.msgprint("OCR parsing failed. Please check the image quality or try a different passport scan.")
                    return

                return frappe._dict(parsed_data), raw
        except Exception as e:
            frappe.msgprint(f'Error in file. {e}')
    

def parse_ocr_data(raw):
    if not raw:
        return

    frappe.errprint(f"Passed in RAW: {raw}")

    # Split and sanitize lines
    lines = raw.splitlines()
    lines = [line.strip() for line in lines if len(line.strip()) > 0]

    # MRZ line matcher
    def is_mrz_line(line):
        return len(line.strip()) >= 40 and re.match(r'^[A-Z0-9<]+$', line.strip()) is not None

    # Try to detect two consecutive MRZ lines
    line1, line2 = None, None
    for i in range(len(lines) - 1):
        if is_mrz_line(lines[i]) and is_mrz_line(lines[i + 1]):
            line1 = lines[i]
            line2 = lines[i + 1]
            break

    if not line1 or not line2:
        frappe.msgprint("Could not detect MRZ lines from OCR output. Please upload a clearer image.")
        return

    try:
        # Passport Type (first character of line1)
        passport_type = line1[0]

        # Nationality (3-letter ISO in line2 at index 10–13)
        nationality = line2[10:13]

        # Last Name: between start and '<<'
        # last_name = line1.split("<<")[0][1:]  # skip 1st char (type)
        last_name = line1.split("<<")[0][1:].split(nationality)[-1]  # Remove 'P<' and take until '<<'


        # First Name: between first and second '<' group after '<<'
        first_name = line1.split("<<")[1].split("<")[0]

        # Passport Number: first 9 chars of line2
        passport_number = line2[:9]

        # Date of Birth (YYMMDD)
        dob_str = line2[13:19]
        try:
            dob = datetime.strptime(dob_str, "%y%m%d").strftime("%d-%m-%Y")
        except:
            dob = ''

        # Sex: 1 char at index 20
        sex = line2[20]

        # Expiry Date (YYMMDD)
        expiry_str = line2[21:27]
        try:
            expiry = datetime.strptime(expiry_str, "%y%m%d").strftime("%d-%m-%Y")
        except:
            expiry = ''

        # CNIC (if applicable): fallback field — adjust as per local layout
        cnic = line2[28:41] if len(line2) >= 41 else ''

        parsed_data = {
            "passport_type": passport_type,
            "last_name": last_name,
            "first_name": first_name,
            "passport_number": passport_number,
            "nationality": nationality,
            "date_of_birth": getdate(dob) if dob else None,
            "sex": sex,
            "date_of_expiry": getdate(expiry) if expiry else None,
            "cnic": cnic
        }

        return parsed_data

    except Exception as e:
        frappe.msgprint(
            f"Error parsing OCR data: {e}. "
            "Please upload a clearer image and verify that the cropped area is correctly selected."
        )
