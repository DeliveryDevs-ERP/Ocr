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
                    
                    # First try MRZ parsing, then enhance with Gemini
                    mrz_data = parse_ocr_data(text)
                    if mrz_data:
                        enhanced_data = self.enhance_with_gemini(mrz_data, text)
                        self.scanned_contents = str(enhanced_data) if enhanced_data else str(mrz_data)
                    else:
                        # Fallback to direct Gemini parsing
                        parsed_data = self.parse_with_gemini(text)
                        self.scanned_contents = str(parsed_data) if parsed_data else text
                else:
                    img = cv2.imread(path)
                    new = self.preprocess_image(img)
                    custom_config = r'--oem 3 --psm 6'
                    raw = pytesseract.image_to_string(img, config=custom_config)
                    
                    # First try MRZ parsing, then enhance with Gemini
                    mrz_data = parse_ocr_data(raw)
                    if mrz_data:
                        enhanced_data = self.enhance_with_gemini(mrz_data, raw)
                        self.scanned_contents = str(enhanced_data) if enhanced_data else str(mrz_data)
                    else:
                        # Fallback to direct Gemini parsing
                        parsed_data = self.parse_with_gemini(raw)
                        self.scanned_contents = str(parsed_data) if parsed_data else raw
            except Exception as e:
                frappe.msgprint(f'Error in file. {e}')

    def enhance_with_gemini(self, mrz_data, raw_ocr_text):
        """
        Send MRZ-extracted data + raw OCR text to Gemini to fill missing fields
        """
        try:
            # Configure Gemini API
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                frappe.msgprint("Gemini API key not found in environment variables.")
                return mrz_data
                
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Convert dates back to string format for JSON serialization
            mrz_data_for_prompt = mrz_data.copy()
            date_fields = ['date_of_birth', 'date_of_expiry', 'date_of_issue']
            for field in date_fields:
                if mrz_data_for_prompt.get(field):
                    if isinstance(mrz_data_for_prompt[field], datetime):
                        mrz_data_for_prompt[field] = mrz_data_for_prompt[field].strftime("%d-%m-%Y")
                    elif hasattr(mrz_data_for_prompt[field], 'strftime'):
                        mrz_data_for_prompt[field] = mrz_data_for_prompt[field].strftime("%d-%m-%Y")
            
            prompt = f"""
            I have already extracted basic passport information using MRZ parsing. Please enhance this data by finding missing information from the raw OCR text.
            
            Current extracted data (from MRZ):
            {json.dumps(mrz_data_for_prompt, indent=2)}
            
            Please analyze the raw OCR text below and fill in any missing fields, especially:
            - date_of_issue (if empty)
            - father_name (extract father's name if mentioned)
            - place_of_stay (extract place of stay/address if mentioned)
            
            Keep all existing MRZ data intact and only add/update missing fields.
            
            Return the enhanced JSON object with this structure:
            {{
                "passport_type": "existing or extracted value",
                "last_name": "existing or extracted value",
                "first_name": "existing or extracted value", 
                "passport_number": "existing or extracted value",
                "nationality": "existing or extracted value",
                "date_of_birth": "existing or extracted value (DD-MM-YYYY)",
                "sex": "existing or extracted value",
                "date_of_expiry": "existing or extracted value (DD-MM-YYYY)",
                "date_of_issue": "extract if missing (DD-MM-YYYY)",
                "cnic": "existing or extracted value",
                "father_name": "extract father's name if available",
                "place_of_stay": "extract place of stay/address if available"
            }}
            
            Raw OCR Text to analyze for missing information:
            {raw_ocr_text}
            
            Instructions:
            - Preserve all existing MRZ data
            - Look for date of issue in various formats like "DD MMM YYYY", or with labels like "Date of Issue", "Issue Date", "Issued", etc.
            - Look for father's name which might appear as "Father's Name:", "Father:", "S/O", "Son of", etc.
            - Look for place of stay/address information
            - Handle date formats like "DD MMM YYYY" (e.g., "06 OCT 2023") and convert to DD-MM-YYYY
            - Return only the JSON object, no additional text
            - If a field cannot be determined, keep existing value or use empty string ""
            """
            
            response = model.generate_content(prompt)
            
            # Parse the JSON response
            json_text = response.text.strip()
            # Remove markdown code blocks if present
            if json_text.startswith('```'):
                json_text = json_text.split('```')[1]
                if json_text.startswith('json'):
                    json_text = json_text[4:]
            
            enhanced_data = json.loads(json_text)
            
            # Convert date strings back to date objects for Frappe
            date_fields = ['date_of_birth', 'date_of_expiry', 'date_of_issue']
            for field in date_fields:
                if enhanced_data.get(field):
                    try:
                        enhanced_data[field] = getdate(enhanced_data[field])
                    except:
                        enhanced_data[field] = None
            
            return enhanced_data
            
        except json.JSONDecodeError:
            frappe.msgprint("Error: Gemini returned invalid JSON format. Using MRZ data only.")
            return mrz_data
        except Exception as e:
            frappe.msgprint(f"Error calling Gemini API for enhancement: {str(e)}. Using MRZ data only.")
            return mrz_data

    def parse_with_gemini(self, raw_ocr_text):
        """
        Send raw OCR text to Gemini API for structured parsing (fallback method)
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
                "date_of_issue": "DD-MM-YYYY",
                "cnic": "string (if available)",
                "father_name": "string (if available)",
                "place_of_stay": "string (if available)"
            }}
            
            OCR Text:
            {raw_ocr_text}
            
            Instructions:
            - Extract information from raw data and MRZ (Machine Readable Zone) lines if present
            - Look for date of issue in various formats like "DD MMM YYYY", or with labels like "Date of Issue", "Issue Date", "Issued", etc.
            - Look for father's name which might appear as "Father's Name:", "Father:", "S/O", "Son of", etc.
            - Look for place of stay/address information
            - Identify dates by context - typically issue date comes before expiry date in passport text
            - Handle formats like "DD MMM YYYY" (e.g., "06 OCT 2023") and convert to DD-MM-YYYY
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
            date_fields = ['date_of_birth', 'date_of_expiry', 'date_of_issue']
            for field in date_fields:
                if parsed_data.get(field):
                    try:
                        parsed_data[field] = getdate(parsed_data[field])
                    except:
                        parsed_data[field] = None
            
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
    Updated scan_passport function: MRZ first, then Gemini enhancement
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
                
                # First extract using MRZ parsing
                mrz_data = parse_ocr_data(text)
                if mrz_data:
                    # Enhance with Gemini to fill missing fields
                    enhanced_data = enhance_with_gemini_api(mrz_data, text)
                    return frappe._dict(enhanced_data), text
                else:
                    # Fallback to direct Gemini parsing
                    parsed_data = parse_with_gemini_api(text)
                    return frappe._dict(parsed_data) if parsed_data else None, text
                
            else:
                img = cv2.imread(path)
                custom_config = r'--oem 3 --psm 6'
                raw = pytesseract.image_to_string(img, config=custom_config)
                
                # First extract using MRZ parsing
                mrz_data = parse_ocr_data(raw)
                if mrz_data:
                    # Enhance with Gemini to fill missing fields
                    enhanced_data = enhance_with_gemini_api(mrz_data, raw)
                    return frappe._dict(enhanced_data), raw
                else:
                    # Fallback to direct Gemini parsing
                    parsed_data = parse_with_gemini_api(raw)
                    return frappe._dict(parsed_data) if parsed_data else None, raw
                
        except Exception as e:
            frappe.msgprint(f'Error in file: {e}')
            return None, None
    
    return None, None

def enhance_with_gemini_api(mrz_data, raw_ocr_text):
    """
    Standalone function to enhance MRZ data with Gemini API
    """
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            frappe.msgprint("Gemini API key not found. Using MRZ data only.")
            return mrz_data
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Convert dates to string format for JSON serialization
        mrz_data_for_prompt = mrz_data.copy()
        date_fields = ['date_of_birth', 'date_of_expiry', 'date_of_issue']
        for field in date_fields:
            if mrz_data_for_prompt.get(field):
                if isinstance(mrz_data_for_prompt[field], datetime):
                    mrz_data_for_prompt[field] = mrz_data_for_prompt[field].strftime("%d-%m-%Y")
                elif hasattr(mrz_data_for_prompt[field], 'strftime'):
                    mrz_data_for_prompt[field] = mrz_data_for_prompt[field].strftime("%d-%m-%Y")
        
        prompt = f"""
        I have already extracted passport information using MRZ (Machine Readable Zone) parsing. 
        Please enhance this data by analyzing the raw OCR text to find missing information.
        
        Current MRZ-extracted data:
        {json.dumps(mrz_data_for_prompt, indent=2)}
        
        Please find and add these missing fields from the raw OCR text:
        1. date_of_issue (if missing or empty)
        2. father_name (look for patterns like "Father's Name:", "Father:", "S/O", "Son of", etc.)
        3. place_of_stay (look for address/place of stay information)
        
        IMPORTANT: Keep ALL existing MRZ data intact. Only add/update missing fields.
        
        Return the enhanced JSON object with this exact structure:
        {{
            "passport_type": "preserve existing value",
            "last_name": "preserve existing value",
            "first_name": "preserve existing value", 
            "passport_number": "preserve existing value",
            "nationality": "preserve existing value",
            "date_of_birth": "preserve existing value (DD-MM-YYYY)",
            "sex": "preserve existing value",
            "date_of_expiry": "preserve existing value (DD-MM-YYYY)",
            "date_of_issue": "find if missing (DD-MM-YYYY)",
            "cnic": "preserve existing value",
            "father_name": "extract if available",
            "place_of_stay": "extract if available"
        }}
        
        Raw OCR Text to search for missing information:
        {raw_ocr_text}
        
        Guidelines for extraction:
        - Look for date of issue in formats: "DD MMM YYYY", "DD-MM-YYYY", "DD/MM/YYYY"
        - Date of issue labels: "Date of Issue", "Issue Date", "Issued", "Date of Issuance"
        - Father's name patterns: "Father's Name:", "Father:", "S/O", "Son of", "Father Name"
        - Place of stay patterns: "Place of Stay", "Address", "Residence", location information
        - Convert dates to DD-MM-YYYY format
        - Return only the JSON object, no additional text
        """
        
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        
        if json_text.startswith('```'):
            lines = json_text.split('\n')
            json_text = '\n'.join(lines[1:-1])
        
        if json_text.startswith('json'):
            json_text = json_text[4:].strip()
            
        enhanced_data = json.loads(json_text)
        
        # Convert date strings back to date objects for Frappe
        date_fields = ['date_of_birth', 'date_of_expiry', 'date_of_issue']
        for field in date_fields:
            if enhanced_data.get(field):
                try:
                    enhanced_data[field] = getdate(enhanced_data[field])
                except:
                    enhanced_data[field] = None
        
        return enhanced_data
        
    except json.JSONDecodeError as e:
        frappe.msgprint(f"Error parsing Gemini enhancement response: {str(e)}. Using MRZ data only.")
        return mrz_data
    except Exception as e:
        frappe.msgprint(f"Error calling Gemini API for enhancement: {str(e)}. Using MRZ data only.")
        return mrz_data

def parse_with_gemini_api(raw_ocr_text):
    """
    Standalone function to parse OCR text with Gemini API (fallback method)
    """
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            frappe.msgprint("Gemini API key not found. Please set GEMINI_API_KEY in your environment variables.")
            return None
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        Extract passport information from the following OCR text and return it as a structured JSON object.
        
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
            "date_of_issue": "",
            "cnic": "",
            "father_name": "",
            "place_of_stay": ""
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
        
        # Convert date strings to date objects for Frappe
        date_fields = ['date_of_birth', 'date_of_expiry', 'date_of_issue']
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
    Enhanced MRZ parsing function with better date_of_issue extraction
    """
    if not raw:
        return None

    lines = raw.splitlines()
    lines = [line.strip() for line in lines if len(line.strip()) > 0]

    def is_mrz_line(line):
        return len(line.strip()) >= 40 and re.match(r'^[A-Za-z0-9<]+$', line.strip()) is not None

    # Extract date of issue from text before MRZ processing
    date_of_issue = None
    all_dates = []
    
    for line in lines:
        # Look for common date of issue patterns with labels
        issue_patterns = [
            r'(?:Date of Issue|Issue Date|Issued|Date of Issuance)[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(?:Date of Issue|Issue Date|Issued|Date of Issuance)[:\s]*(\d{1,2}\s+\w+\s+\d{2,4})',
            r'(?:Date of Issue|Issue Date|Issued|Date of Issuance)[:\s]*(\d{2}\d{2}\d{2})',
        ]
        
        for pattern in issue_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                try:
                    # Try different date formats
                    for fmt in ['%d-%m-%Y', '%d/%m/%Y', '%d-%m-%y', '%d/%m/%y', '%y%m%d', '%d %B %Y', '%d %b %Y']:
                        try:
                            date_of_issue = datetime.strptime(date_str, fmt).strftime("%d-%m-%Y")
                            break
                        except:
                            continue
                    if date_of_issue:
                        break
                except:
                    continue
        if date_of_issue:
            break
    
    # If no labeled date found, look for date patterns in the text
    if not date_of_issue:
        date_patterns = [
            r'(\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4})',  # 06 OCT 2023
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',  # DD-MM-YYYY or DD/MM/YYYY
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2})',  # DD-MM-YY or DD/MM/YY
        ]
        
        for line in lines:
            for pattern in date_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    try:
                        # Try different date formats
                        date_formats = ['%d %b %Y', '%d %B %Y', '%d-%m-%Y', '%d/%m/%Y', '%d-%m-%y', '%d/%m/%y']
                        for fmt in date_formats:
                            try:
                                parsed_date = datetime.strptime(match, fmt)
                                formatted_date = parsed_date.strftime("%d-%m-%Y")
                                all_dates.append((formatted_date, parsed_date))
                                break
                            except:
                                continue
                    except:
                        continue
        
        # If we found dates, try to identify which one might be the issue date
        if all_dates:
            # Sort dates by date value
            all_dates.sort(key=lambda x: x[1])
            
            # Look for a date that's likely to be issue date (usually comes before expiry)
            # In most cases, if we have 2 dates, the earlier one is likely the issue date
            if len(all_dates) >= 2:
                date_of_issue = all_dates[0][0]  # Take the earliest date
            elif len(all_dates) == 1:
                # If only one date found and it's reasonable for issue date, use it
                date_obj = all_dates[0][1]
                current_year = datetime.now().year
                # Issue date should be within reasonable range (not too old, not in future)
                if date_obj.year >= (current_year - 10) and date_obj.year <= current_year:
                    date_of_issue = all_dates[0][0]

    # Find MRZ lines
    line1, line2 = None, None
    for i in range(len(lines) - 1):
        if is_mrz_line(lines[i]) and is_mrz_line(lines[i + 1]):
            line1 = lines[i]
            line2 = lines[i + 1]
            break
        elif line1 is None and lines[i][:2] in ['P<', 'D<', 'S<', 'F<']:
            line1 = lines[i]
            if i + 1 < len(lines):
                line2 = lines[i + 1]
            break

    if not line1 or not line2:
        frappe.msgprint("Could not detect MRZ lines from OCR output. Will try Gemini fallback.")
        return None

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
            "date_of_issue": getdate(date_of_issue) if date_of_issue else None,
            "cnic": cnic,
            "father_name": "",  # Will be filled by Gemini
            "place_of_stay": ""  # Will be filled by Gemini
        }

        return parsed_data

    except Exception as e:
        frappe.msgprint(f"Error parsing MRZ data: {e}. Will try Gemini fallback.")
        return None