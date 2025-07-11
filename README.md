# 🔍 Frappe OCR – AI‑Powered Document Scanner

<div align="center">

![Frappe OCR Banner](https://img.shields.io/badge/Frappe-OCR-blue?style=for-the-badge\&logo=python\&logoColor=white)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![Frappe Framework](https://img.shields.io/badge/Frappe-Framework-orange?style=flat-square)](https://frappeframework.com/)
[![Tesseract OCR](https://img.shields.io/badge/Tesseract-OCR-red?style=flat-square)](https://github.com/tesseract-ocr/tesseract)
[![Gemini AI](https://img.shields.io/badge/Gemini-AI-purple?style=flat-square)](https://ai.google.dev/)

*Transform your document scanning workflow with AI‑powered OCR technology.*

[Features](#✨-features) • [Installation](#🛠-installation) • [Usage](#📖-usage) • [API](#🔌-api-reference) • [Contributing](#🤝-contributing)

</div>

---

## 🚀 What is Frappe OCR?

Frappe OCR is a passport‑scanning and text‑extraction app for the Frappe Framework. It combines traditional OCR (Tesseract) with Google’s Gemini AI for accurate, intelligent document processing.

### 🎯 Ideal for:

* **Government Services** – Passport applications and renewals
* **Travel Agencies** – Fast customer data entry from travel documents
* **Banks & Financial Institutions** – KYC document processing
* **Immigration Services** – Automated document verification
* **Any Business** – General document digitization

---

## ✨ Features

### 🧠 AI‑Powered Intelligence

* **Gemini AI Integration** – Intelligent parsing and error correction
* **Smart Error Correction** – Automatically fixes common OCR mistakes
* **Contextual Understanding** – Extracts structured data (e.g., MRZ fields)

### 📄 Document Support

* **Multi‑format** – PNG, JPG, JPEG, PDF
* **Passport‑Optimized** – MRZ (Machine Readable Zone) parsing
* **Batch Processing** – Scan multiple files at once

### 🔧 Advanced OCR Pipeline

* **Image Preprocessing** – Enhancement and noise reduction
* **Adaptive Thresholding** – Robust to varying lighting
* **Skew Correction** – Auto‑straighten tilted scans

### 💾 Frappe Integration

* **Native DocTypes** – Leverage File Manager with full Frappe features
* **Role‑Based Access** – Fine‑grained permissions
* **Search & Filter** – Full‑text search on extracted content
* **RESTful API** – Integrate with external systems

---

## 🛠 Installation

### Prerequisites

Install Tesseract and development headers:

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install tesseract-ocr libtesseract-dev

# macOS (Homebrew)
brew install tesseract

# CentOS/RHEL
sudo yum install tesseract tesseract-devel
```

### Install the App

```bash
# In your bench directory
cd /path/to/bench

# Clone and install
bench get-app https://github.com/DeliveryDevs-ERP/ocr.git
bench --site your-site install-app ocr

# Install Python dependencies
bench setup requirements

# Restart bench
bench restart
```

### Configuration

Create a `.env` file at the app root:

```bash
# Gemini AI settings
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

> 🔑 **Get your Gemini API Key:** Visit [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## 📖 Usage

### 🖥 Web Interface

1. Open the **OCR** module in your Frappe Desk.
2. Click **New** to create a File Manager record.
3. Upload an image or PDF.
4. Click **Save** – OCR runs automatically.
5. Review extracted text in **Scanned Contents**.

### 📱 API Usage

```python
import frappe

# Scan a passport image
parsed, raw = frappe.call(
    'ocr.ocr.doctype.file_manager.file_manager.scan_passport',
    file_url='/files/passport_scan.jpg'
)
print(parsed)
print(raw)
```

#### Sample Response

```json
{
  "passport_type": "P",
  "last_name": "SMITH",
  "first_name": "JOHN",
  "passport_number": "AB1234567",
  "nationality": "USA",
  "date_of_birth": "1985-06-15",
  "sex": "M",
  "date_of_expiry": "2030-12-20",
  "cnic": "1234567890123"
}
```

---

## 🎮 Demo

### Before OCR ➡️ After OCR

| Raw Scan                                                                                                                                    |                                   Extracted Data                                   |
| :------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------: |
| ![Passport Scan](https://en.wikipedia.org/wiki/United_States_passport#/media/File:United_States_Next_Generation_Passport_signature_and_biodata_page.jpg) | `json{"first_name": "JOHN", "last_name": "SMITH", "passport_number": "AB1234567"}` |

---

## 🔌 API Reference

### `scan_passport(file_url)`

Extracts structured data from passport files.

**Parameters:**

* `file_url` (string): Path to the uploaded file

**Returns:**

* `parsed_data` (dict): Structured fields
* `raw_text` (string): Full OCR output

---

## 🤝 Contributing

1. Fork this repo
2. Create a feature branch (`git checkout -b feature/XYZ`)
3. Commit your changes (`git commit -m "feat: add XYZ"`)
4. Push to branch (`git push origin feature/XYZ`)
5. Open a Pull Request

We welcome contributions of all kinds! Feel free to open issues or submit PRs.

---

---

*© 2025 Deliverydevs. Licensed under MIT.*
