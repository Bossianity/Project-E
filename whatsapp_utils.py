import os
import json
import time
import random
import logging
import requests

WASENDER_API_URL = os.getenv('WASENDER_API_URL', "https://www.wasenderapi.com/api/send-message")
WASENDER_API_TOKEN = os.getenv('WASENDER_API_TOKEN')
HTTP_SESSION = requests.Session()

# Placeholder for send_whapi_request
def send_whapi_request(endpoint, payload):
    WHAPI_BASE_URL = os.getenv('WHAPI_BASE_URL', 'https://api.whapi.com/v1/') # Default if not set
    full_url = f"{WHAPI_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    api_token = os.getenv("WHAPI_API_TOKEN", WASENDER_API_TOKEN) # Fallback to WASENDER_API_TOKEN

    if not api_token:
        logging.error(f"No API token for Whapi. Set WHAPI_API_TOKEN or WASENDER_API_TOKEN.")
        return {"sent": False, "error": "API token not configured"}

    headers = {
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json'
    }
    logging.info(f"Sending POST to Whapi: {full_url}, Payload: {json.dumps(payload)}")
    try:
        response = HTTP_SESSION.post(full_url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error for {full_url}: {e}. Response: {e.response.text[:500] if e.response else 'No text'}")
        return {"sent": False, "error": str(e), "response_text": e.response.text[:200] if e.response else None}
    except requests.exceptions.RequestException as e:
        logging.error(f"Request exception for {full_url}: {e}")
        return {"sent": False, "error": str(e)}

def send_interactive_button_message(to, message_data):
    endpoint = 'messages/interactive'
    buttons_payload = []
    for btn in message_data.get('buttons', []):
        buttons_payload.append({
            "type": "quick_reply",
            "title": btn.get('title'),
            "id": btn.get('id')
        })

    payload = {
        "to": to,
        "type": "button",
        "header": {"text": message_data.get('header', '')},
        "body": {"text": message_data.get('body', '')},
        "footer": {"text": message_data.get('footer', '')},
        "action": {"buttons": buttons_payload},
        "view_once": False
    }

    if not payload["header"]["text"]: del payload["header"]
    if not payload["footer"]["text"]: del payload["footer"]
    if not payload["action"]["buttons"]:
        logging.error(f"Interactive message to {to} has no buttons.")
        return False

    logging.info(f"Sending interactive message to {to}. Payload: {json.dumps(payload, indent=2)}")
    response = send_whapi_request(endpoint, payload)

    if response and response.get('sent'):
        logging.info(f"Successfully sent interactive message to {to}.")
        return True
    else:
        logging.error(f"Failed to send interactive message to {to}. Response: {response}")
        return False

def send_whatsapp_message(to, text):
    if not all([WASENDER_API_URL, WASENDER_API_TOKEN, HTTP_SESSION]):
        logging.error("WASender API not configured.")
        return False

    clean_to = to.split('@')[0] if "@s.whatsapp.net" in to else to
    payload = {'to': clean_to, 'text': text}
    headers = {'Authorization': f'Bearer {WASENDER_API_TOKEN}', 'Content-Type': 'application/json'}
    max_retries = 4
    for attempt in range(max_retries):
        try:
            logging.info(f"Sending to {clean_to} (Attempt {attempt+1}/{max_retries}). Text: {text[:50]}...")
            resp = HTTP_SESSION.post(WASENDER_API_URL, json=payload, headers=headers, timeout=15)
            if not (200 <= resp.status_code < 300):
                logging.error(f"Error sending to {clean_to} (Attempt {attempt+1}). Status: {resp.status_code}. Resp: {resp.text[:500]}")
                if resp.status_code == 401: return False
            else:
                data = resp.json()
                if data.get("success") is True:
                    logging.info(f"Sent message to {clean_to}: {text[:50]}...")
                    return True
                else:
                    logging.warning(f"API success false for {clean_to}. JSON: {data}")
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout sending to {clean_to} (Attempt {attempt+1}).")
        except requests.exceptions.RequestException as e_req:
            logging.warning(f"RequestException for {clean_to} (Attempt {attempt+1}): {e_req}")
        if attempt < max_retries - 1:
            time.sleep((2 ** attempt) + random.uniform(0.1, 0.9))
    logging.error(f"All attempts failed for {clean_to}. Message: {text[:50]}...")
    return False

def send_whatsapp_image_message(to, caption, image_url):
    if not all([WASENDER_API_URL, WASENDER_API_TOKEN, HTTP_SESSION]):
        logging.error("WASender API not configured for image.")
        return False

    clean_to = to.split('@')[0] if "@s.whatsapp.net" in to else to
    payload = {'to': clean_to, 'imageUrl': image_url}
    if caption and isinstance(caption, str) and caption.strip():
        payload['text'] = caption.strip()
    headers = {'Authorization': f'Bearer {WASENDER_API_TOKEN}', 'Content-Type': 'application/json'}
    max_retries = 4
    for attempt in range(max_retries):
        try:
            logging.info(f"Sending image to {clean_to} (Attempt {attempt+1}). URL: {image_url}")
            resp = HTTP_SESSION.post(WASENDER_API_URL, json=payload, headers=headers, timeout=20)
            if not (200 <= resp.status_code < 300):
                logging.error(f"Error sending image to {clean_to} (Attempt {attempt+1}). Status: {resp.status_code}. Resp: {resp.text[:500]}")
                if resp.status_code == 401: return False
            else:
                data = resp.json()
                if data.get("success") is True:
                    logging.info(f"Sent image to {clean_to}. URL: {image_url}")
                    return True
                else:
                    logging.warning(f"API success false for image to {clean_to}. JSON: {data}")
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout sending image to {clean_to} (Attempt {attempt+1}).")
        except requests.exceptions.RequestException as e:
            logging.warning(f"RequestException for image to {clean_to} (Attempt {attempt+1}): {e}")
        if attempt < max_retries - 1:
            time.sleep((2 ** attempt) + random.uniform(0.1, 0.9))
    logging.error(f"All image attempts failed for {clean_to}. URL: {image_url}")
    return False
