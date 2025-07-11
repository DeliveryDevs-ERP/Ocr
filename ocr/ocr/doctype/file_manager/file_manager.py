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
import google.generativeai as genai
import json
from dotenv import load_dotenv

load_dotenv()

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
                    new = self.preprocess_image(img)
                    custom_config = r'--oem 3 --psm 6'
                    raw = pytesseract.image_to_string(img, config=custom_config)
                    
                    # Use Gemini API to parse OCR data
                    parsed_data = self.parse_with_gemini(raw)
                    self.scanned_contents = str(parsed_data) if parsed_data else raw
            except Exception as e:
                frappe.msgprint(f'Error in file. {e}')

    def parse_with_gemini(self, raw_ocr_text):
        """
        Send raw OCR text to Gemini API for structured parsing
        """
        try:
            # Configure Gemini API
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                frappe.msgprint("Gemini API key not found in environment variables.")
                return None
                
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            Extract passport information from the following OCR text and return it as a JSON object.
            The OCR text may contain errors, noise, or formatting issues. Please extract the most likely correct values.
            
            Required JSON format:
            {{
                "passport_type": "P",
                "last_name": "string",
                "first_name": "string", 
                "passport_number": "string",
                "nationality": "string (3-letter ISO code)",
                "date_of_birth": "DD-MM-YYYY",
                "sex": "M or F",
                "date_of_expiry": "DD-MM-YYYY",
                "cnic": "string (if available)"
            }}
            
            OCR Text:
            {raw_ocr_text}
            
            Instructions:
            - Extract information from raw data and MRZ (Machine Readable Zone) lines if present
            - If dates are in YYMMDD format, convert to DD-MM-YYYY
            - Clean up any OCR errors in names and numbers
            - Return only the JSON object, no additional text
            - If a field cannot be determined, use empty string ""
            """
            
            response = model.generate_content(prompt)
            
            # Parse the JSON response
            json_text = response.text.strip()
            # Remove markdown code blocks if present
            if json_text.startswith('```'):
                json_text = json_text.split('```')[1]
                if json_text.startswith('json'):
                    json_text = json_text[4:]
            
            parsed_data = json.loads(json_text)
            
            # Convert date strings to date objects for Frappe
            if parsed_data.get('date_of_birth'):
                try:
                    parsed_data['date_of_birth'] = getdate(parsed_data['date_of_birth'])
                except:
                    parsed_data['date_of_birth'] = None
                    
            if parsed_data.get('date_of_expiry'):
                try:
                    parsed_data['date_of_expiry'] = getdate(parsed_data['date_of_expiry'])
                except:
                    parsed_data['date_of_expiry'] = None
            
            return parsed_data
            
        except json.JSONDecodeError:
            frappe.msgprint("Error: Gemini returned invalid JSON format.")
            return None
        except Exception as e:
            frappe.msgprint(f"Error calling Gemini API: {str(e)}")
            return None

    # ... (keep all your existing image processing methods)
    
    def get_grayscale(self,image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, image):
        return cv2.medianBlur(image, 5)

    def thresholding(self, image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def dilate(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    def erode(self,image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    def opening(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def canny(self, image):
        return cv2.Canny(image, 100, 200)

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

    def match_template(self, image, template):
        return cv2.matchTemplate(image, template,cv2.TM_CCOEFF_NORMED)
    
    def sharpen_image(self, image):
        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def convert_to_black_and_white(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw_image

    def reduce_blur(self, image):
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        unsharp_image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        return unsharp_image

    def preprocess_image(self, image):
        sharpened = self.sharpen_image(image)
        final_image = self.reduce_blur(sharpened)
        return final_image

@frappe.whitelist()
def scan_passport(file_url):
    """
    Updated scan_passport function using Gemini API
    """
    path = os.getcwd()
    site = cstr(frappe.local.site)
    path = f'{path}/{site}{file_url}'
    
    if os.path.isfile(path):
        try:
            file_extension = os.path.splitext(path)[1]
            
            if file_extension.lower() in {'.pdf'}:
                doc = convert_from_path(path)
                text = ""
                for page in doc:
                    text += pytesseract.image_to_string(page)
                
                # Use Gemini for PDF text parsing too
                try:
                    parsed_data = parse_with_gemini_api(text)
                except:
                    parsed_data = parse_ocr_data(text)
                return parsed_data, text
                
            else:
                img = cv2.imread(path)
                custom_config = r'--oem 3 --psm 6'
                raw = pytesseract.image_to_string(img, config=custom_config)
                
                try:
                    parsed_data = parse_with_gemini_api(raw)
                except:
                    parsed_data = parse_ocr_data(raw)
                
                if not parsed_data:
                    frappe.msgprint("OCR parsing failed. Please check the image quality or try a different passport scan.")
                    return None, raw

                return frappe._dict(parsed_data), raw
                
        except Exception as e:
            frappe.msgprint(f'Error in file: {e}')
            return None, None
    
    return None, None

def parse_with_gemini_api(raw_ocr_text):
    """
    Standalone function to parse OCR text with Gemini API
    """
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            frappe.msgprint("Gemini API key not found. Please set GEMINI_API_KEY in your environment variables.")
            return None
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Enhanced prompt for better passport data extraction
        prompt = f"""
        You are an expert at extracting passport information from OCR text. 
        Analyze the following OCR text and extract passport data into a structured JSON format.
        
        The text may contain OCR errors, extra characters, or formatting issues. Please:
        1. Identify MRZ (Machine Readable Zone) lines if present
        2. Extract accurate information despite OCR errors
        3. Convert dates from YYMMDD to DD-MM-YYYY format
        4. Clean up passport numbers and remove extra characters
        5. Extract 3-letter nationality codes (ISO format)
        
        Return ONLY a JSON object with this exact structure:
        {{
            "passport_type": "",
            "last_name": "",
            "first_name": "", 
            "passport_number": "",
            "nationality": "",
            "date_of_birth": "",
            "sex": "",
            "date_of_expiry": "",
            "cnic": ""
        }}
        
        OCR Text to analyze:
        {raw_ocr_text}
        
        Important: Return only the JSON object, no additional text or formatting.
        """
        
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        
        if json_text.startswith('```'):
            lines = json_text.split('\n')
            json_text = '\n'.join(lines[1:-1])
        
        if json_text.startswith('json'):
            json_text = json_text[4:].strip()
        parsed_data = json.loads(json_text)
        date_fields = ['date_of_birth', 'date_of_expiry']
        for field in date_fields:
            if parsed_data.get(field):
                try:
                    parsed_data[field] = getdate(parsed_data[field])
                except:
                    parsed_data[field] = None
        
        return parsed_data
        
    except json.JSONDecodeError as e:
        frappe.msgprint(f"Error parsing Gemini response as JSON: {str(e)}")
        return None
    except Exception as e:
        frappe.msgprint(f"Error calling Gemini API: {str(e)}")
        return None

def parse_ocr_data(raw):
    """
    Original parsing function - kept as fallback
    """
    if not raw:
        return

    lines = raw.splitlines()
    lines = [line.strip() for line in lines if len(line.strip()) > 0]

    def is_mrz_line(line):
        return len(line.strip()) >= 40 and re.match(r'^[A-Za-z0-9<]+$', line.strip()) is not None

    line1, line2 = None, None
    for i in range(len(lines) - 1):
        if is_mrz_line(lines[i]) and is_mrz_line(lines[i + 1]):
            line1 = lines[i]
            line2 = lines[i + 1]
        elif line1 is None and lines[i][:2] in ['P<', 'D<', 'S<', 'F<']:
            line1 = lines[i]
            line2 = lines[i + 1]
            break

    if not line1 or not line2:
        frappe.msgprint("Could not detect MRZ lines from OCR output. Please upload a clearer image.")
        return

    try:
        passport_type = line1[0]
        nationality = line2[10:13]
        last_name = line1.split("<<")[0][1:].split(nationality)[-1]
        first_name = line1.split("<<")[1].split("<")[0]
        passport_number_raw = line2[:9]
        passport_number = ''.join(filter(str.isalnum, passport_number_raw))

        dob_str = line2[13:19]
        try:
            dob = datetime.strptime(dob_str, "%y%m%d").strftime("%d-%m-%Y")
        except:
            dob = ''

        sex = line2[20]

        expiry_str = line2[21:27]
        try:
            expiry = datetime.strptime(expiry_str, "%y%m%d").strftime("%d-%m-%Y")
        except:
            expiry = ''

        cnic_raw = line2[28:41] if len(line2) >= 41 else ''
        cnic = ''.join(filter(str.isdigit, cnic_raw))

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