import os
import time
import logging
import json
from datetime import datetime
import pytz

from whatsapp_utils import send_whatsapp_message, send_interactive_button_message

# Placeholder/Assumed Functions (definitions needed elsewhere in the actual project)
def get_google_sheets_service():
    logging.warning("Placeholder: get_google_sheets_service() called. Returning None.")
    return None

def read_sheet_data(sheets_service, sheet_id, contacts_sheet_name):
    logging.warning(f"Placeholder: read_sheet_data({sheet_id}, {contacts_sheet_name}) called. Returning minimal dummy data.")
    header_map = {'PhoneNumber': 0, 'ClientName': 1, 'MessageStatus': 2, 'InterestedService': 3, 'LastContactedDate': 4}
    dummy_rows = [
        {'data': {'PhoneNumber': '1234567890', 'ClientName': 'Test Client', 'MessageStatus': '', 'InterestedService': 'Testing'}, 'original_row_index': 2}
    ]
    return dummy_rows, header_map

def update_cell_value(sheets_service, sheet_id, contacts_sheet_name, original_row_idx_1_based, col_header, new_status):
    logging.warning(f"Placeholder: update_cell_value({sheet_id}, {original_row_idx_1_based}, {col_header}, {new_status}) called.")
    return True

def get_message_template_from_sheet(sheets_service, sheet_id, template_sheet_name):
    logging.warning(f"Placeholder: get_message_template_from_sheet({sheet_id}, {template_sheet_name}) called. Returning dummy template.")
    dummy_interactive_template = {
        "header": "Test Header",
        "body": "Hello {{ClientName}}, interested in {{ServiceName}}?",
        "footer": "Test Footer",
        "buttons": [{"title": "Yes", "id": "yes_id"}, {"title": "No", "id": "no_id"}]
    }
    return dummy_interactive_template, "Simple hello {{ClientName}}", True

def personalize_interactive_message_data(template_data, placeholders):
    logging.warning(f"Placeholder: personalize_interactive_message_data with {placeholders} called.")
    personalized_data = json.loads(json.dumps(template_data)) # Deep copy
    personalized_data['body'] = template_data['body'].replace("{{ClientName}}", placeholders.get('ClientName', 'Valued Customer'))
    personalized_data['body'] = personalized_data['body'].replace("{{ServiceName}}", placeholders.get('ServiceName', 'our services'))
    return personalized_data

def _execute_campaign_logic(sheet_id, agent_sender_id):
    logging.info(f"Executing campaign logic for Sheet ID: {sheet_id}, by {agent_sender_id}.")
    sheets_service = get_google_sheets_service()

    template_sheet_name = os.getenv('MESSAGE_TEMPLATE_SHEET_NAME', 'MessageTemplate')
    contacts_sheet_name = os.getenv('CONTACTS_SHEET_NAME', 'Sheet1')
    delay_seconds = int(os.getenv('OUTREACH_MESSAGE_DELAY_SECONDS', 5))
    dubai_tz = pytz.timezone('Asia/Dubai')

    try:
        interactive_template, simple_template, is_interactive = get_message_template_from_sheet(sheets_service, sheet_id, template_sheet_name)
        if not is_interactive and not simple_template:
             logging.error(f"No message templates loaded from {template_sheet_name}.")
             return
    except Exception as e:
        logging.error(f"Failed to get message template: {e}", exc_info=True)
        return

    rows_data, header_map = read_sheet_data(sheets_service, sheet_id, contacts_sheet_name)
    if not rows_data:
        logging.error(f"No contact data from {contacts_sheet_name}.")
        return

    sent_count, failed_count, skipped_count = 0, 0, 0
    for row_info in rows_data:
        row_values_dict = row_info['data']
        original_row_idx_1_based = row_info['original_row_index']

        phone_number_raw = row_values_dict.get('PhoneNumber')
        if not phone_number_raw:
            skipped_count += 1
            continue

        phone_number_str = str(phone_number_raw).strip()
        if '@s.whatsapp.net' not in phone_number_str:
            cleaned_number = ''.join(filter(str.isdigit, phone_number_str))
            if not cleaned_number:
                skipped_count += 1
                continue
            formatted_phone_number = f"{cleaned_number}@s.whatsapp.net"
        else:
            formatted_phone_number = phone_number_str

        current_status = str(row_values_dict.get('MessageStatus', '')).strip().lower()
        if current_status in ["sent", "replied", "completed", "success"]:
            skipped_count += 1
            continue

        placeholders = {
            'ClientName': str(row_values_dict.get('ClientName', 'Valued Customer')).strip(),
            'ServiceName': str(row_values_dict.get('InterestedService', 'our services')).strip()
        }

        message_sent = False
        if is_interactive and interactive_template.get('buttons'):
            personalized_data = personalize_interactive_message_data(interactive_template, placeholders)
            logging.info(f"Row {original_row_idx_1_based}: Sending INTERACTIVE to {formatted_phone_number}.")
            message_sent = send_interactive_button_message(formatted_phone_number, personalized_data)
        elif simple_template:
            personalized_message = simple_template.replace("{{ClientName}}", placeholders['ClientName']).replace("{{ServiceName}}", placeholders['ServiceName'])
            logging.info(f"Row {original_row_idx_1_based}: Sending SIMPLE to {formatted_phone_number}.")
            message_sent = send_whatsapp_message(formatted_phone_number, personalized_message)
        else:
            skipped_count +=1
            continue

        new_status = "Sent" if message_sent else "Failed - API Error"
        if 'MessageStatus' in header_map: # Ensure sheets_service is checked if actual update is intended
            update_cell_value(sheets_service, sheet_id, contacts_sheet_name, original_row_idx_1_based, header_map['MessageStatus'], new_status)

        if message_sent: sent_count += 1
        else: failed_count += 1

        if 'LastContactedDate' in header_map: # Ensure sheets_service is checked
            timestamp = datetime.now(dubai_tz).strftime("%Y-%m-%d %H:%M:%S")
            update_cell_value(sheets_service, sheet_id, contacts_sheet_name, original_row_idx_1_based, header_map['LastContactedDate'], timestamp)

        time.sleep(delay_seconds)

    summary_message = f"Campaign {sheet_id}: Sent {sent_count}, Failed {failed_count}, Skipped {skipped_count}"
    logging.info(summary_message)
    if agent_sender_id:
         send_whatsapp_message(agent_sender_id, summary_message)
    logging.info(f"Campaign {sheet_id} finished.")

def process_outreach_campaign(sheet_id, agent_sender_id, app_context=None):
    if app_context:
        with app_context: # Use Flask app context if available
            _execute_campaign_logic(sheet_id, agent_sender_id)
    else:
        # Run directly if no app_context (e.g., for testing)
        _execute_campaign_logic(sheet_id, agent_sender_id)
