import os
import requests
import json
import re
import pandas as pd
import aiohttp
import asyncio
from urllib.parse import quote
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import datetime
from pytz import timezone
from dotenv import load_dotenv
import logging
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# === Configuration Section ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Salesforce Configuration
client_id = os.getenv('SF_CLIENT_ID')
client_secret = os.getenv('SF_CLIENT_SECRET')
username = os.getenv('SF_USERNAME')
password = os.getenv('SF_PASSWORD')
login_url = os.getenv('SF_LOGIN_URL')
SF_LEADS_URL = "https://waveinfratech.my.salesforce.com/services/data/v58.0/query/?q="
SF_USERS_URL = "https://waveinfratech.my.salesforce.com/services/data/v61.0/query/?q="
SF_CASES_URL = "https://waveinfratech.my.salesforce.com/services/data/v58.0/query/?q="
SF_EVENTS_URL = "https://waveinfratech.my.salesforce.com/services/data/v58.0/query/?q="
SF_OPPORTUNITIES_URL = "https://waveinfratech.my.salesforce.com/services/data/v58.0/query/?q="
SF_TASKS_URL = "https://waveinfratech.my.salesforce.com/services/data/v62.0/query/?q="

# WatsonX Configuration
WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_URL = os.getenv('WATSONX_URL')
WATSONX_MODEL_ID = os.getenv('WATSONX_MODEL_ID')
IBM_CLOUD_IAM_URL = os.getenv('IBM_CLOUD_IAM_URL')

LEAD_QUERY_LIMIT = 30000
CASE_QUERY_LIMIT = 30000
EVENT_QUERY_LIMIT = 30000

# Field Mappings
LEAD_FIELD_MAPPING = {
    'id': 'Id', 'lead_id__c': 'Lead_Id__c', 'customer_feedback__c': 'Customer_Feedback__c', 'city__c': 'City__c',
    'leadsource': 'LeadSource', 'lead_source_sub_category__c': 'Lead_Source_Sub_Category__c', 'project_category__c': 'Project_Category__c',
    'project__c': 'Project__c', 'property_size__c': 'Property_Size__c', 'property_type__c': 'Property_Type__c', 'budget_range__c': 'Budget_Range__c',
    'rating': 'Rating', 'status': 'Status', 'Contact_Medium__c': 'Contact_Medium__c', 'ownerid': 'OwnerId', 'createddate': 'CreatedDate',
    'open_lead_reasons__c': 'Open_Lead_reasons__c', 'junk_reason__c': 'Junk_Reason__c', 'disqualification_date__c': 'Disqualification_Date__c',
    'disqualification_reason__c': 'Disqualification_Reason__c', 'disqualified_date_time__c': 'Disqualified_Date_Time__c',
    'Transfer_Status__c': 'Transfer_Status__c', 'is_appointment_booked__c': 'Is_Appointment_Booked__c', 'lead_converted__c': 'Lead_Converted__c'
}

USER_FIELD_MAPPING = {
    'id': 'Id', 'first_name': 'FirstName', 'firstname': 'FirstName', 'last_name': 'LastName', 'lastname': 'LastName',
    'name': 'Name', 'email': 'Email', 'Department': 'Department', 'username': 'Username', 'profile': 'ProfileId',
    'profileid': 'ProfileId', 'role': 'UserRoleId', 'userroleid': 'UserRoleId', 'is_active': 'IsActive', 'isactive': 'IsActive',
    'created_date': 'CreatedDate', 'createddate': 'CreatedDate', 'last_login_date': 'LastLoginDate', 'lastlogindate': 'LastLoginDate'
}

CASE_FIELD_MAPPING = {
   'id': 'Id',
    'action_taken__c': 'Action_Taken__c',
    'back_office_date__c': 'Back_Office_Date__c',
    'back_office_remarks__c': 'Back_Office_Remarks__c',
    'corporate_closure_remark__c': 'Corporate_Closure_Remark__c',
    'corporate_closure_by__c': 'Corporate_Closure_by__c',
    'corporate_closure_date__c': 'Corporate_Closure_date__c',
    'description': 'Description',
    'feedback__c': 'Feedback__c',
    'first_assigned_by__c': 'First_Assigned_By__c',
    'first_assigned_to__c': 'First_Assigned_To__c',
    'first_assigned_on_date_and_time__c': 'First_Assigned_on_Date_and_Time__c',
    'opportunity__c': 'Opportunity__c',
    'origin': 'Origin',
    're_assigned_by1__c': 'Re_Assigned_By1__c',
    're_assigned_on_date_and_time__c': 'Re_Assigned_On_Date_And_Time__c',
    're_assigned_to__c': 'Re_Assigned_To__c',
    're_open_date_and_time__c': 'Re_Open_Date_and_time__c',
    're_open_by__c': 'Re_Open_by__c',
    'resolved_by__c': 'Resolved_By__c',
    'service_request_number__c': 'Service_Request_Number__c',
    'service_request_type__c': 'Service_Request_Type__c',
    'service_sub_category__c': 'Service_Sub_catogery__c',
    'subject': 'Subject',
    'type': 'Type',
    'createddate': 'CreatedDate'
}

EVENT_FIELD_MAPPING = {
    'id': 'Id', 'account_id': 'LeadId', 'appointment_status__c': 'Lead_Id__c',
 'created_by_id': 'CreatedById', 'created_date': 'C', 'lead__c': 'lead__c',
 'LeadId': 'Lead'}

OPPORTUNITY_FIELD_MAPPING = {
    'id': 'Id',
    'Lead_Id__c': 'Lead_Id__c',
    'Project__c': 'Project__c',
    'Project_Category__c': 'Project_Category__c',
    'CreatedById': 'CreatedById',
    'OwnerId': 'OwnerId',
    'CreatedDate': 'CreatedDate',
    'LeadSource': 'LeadSource',
    'Lead_Source_Sub_Category__c': 'Lead_Source_Sub_Category__c',
    'Lead_Rating_By_Sales__c': 'Lead_Rating_By_Sales__c',
    'Range_Budget__c': 'Range_Budget__c',
    'Property_Type__c': 'Property_Type__c',
    'Property_Size__c': 'Property_Size__c',
    'Sales_Team_Feedback__c': 'Sales_Team_Feedback__c',
    'Sales_Open_Reason__c': 'Sales_Open_Reason__c',
    'Disqualification_Reason__c': 'Disqualification_Reason__c',
    'Disqualified_Date__c': 'Disqualified_Date__c',
    'StageName': 'StageName',
    'SAP_Customer_code__c': 'SAP_Customer_code__c',
    'Registration_Number__c': 'Registration_Number__c',
    'Sales_Order_Number__c': 'Sales_Order_Number__c',
    'Sales_Order_Date__c':  'Sales_Order_Date__c',
    'Age__c': 'Age__c',
    'SAP_City__c': 'SAP_City__c',
    'Country_SAP__c': 'Country_SAP__c',
    'State__c  ': 'State__c  '
}

TASK_FIELD_MAPPING = {
    'id': 'Id', 'lead_id__c': 'Lead_Id__c', 'opp_lead_id__c': 'Opp_Lead_Id__c', 'transfer_status__c': 'Transfer_Status__c',
    'customer_feedback__c': 'Customer_Feedback__c', 'sales_team_feedback__c': 'Sales_Team_Feedback__c', 'status': 'Status',
    'follow_up_status__c': 'Follow_Up_Status__c', 'subject': 'Subject', 'ownerid': 'OwnerId', 'createdbyid': 'CreatedById'
}

TASK_FIELD_DISPLAY_NAMES = {
    'Id': 'Id', 'Lead_Id__c': 'Lead_Id__c', 'Opp_Lead_Id__c': 'Opp_Lead_Id__c', 'Transfer_Status__c': 'Transfer_Status__c',
    'Customer_Feedback__c': 'Customer_Feedback__c', 'Sales_Team_Feedback__c': 'Sales_Team_Feedback__c', 'Status': 'Status',
    'Follow_Up_Status__c': 'Follow_Up_Status__c', 'Subject': 'Subject', 'OwnerId': 'OwnerId', 'CreatedById': 'CreatedById'
}

TASK_FIELD_VALUES = {
    'Status': ['Not Started', 'In Progress', 'Completed', 'Waiting on someone else', 'Deferred'],
    'Follow_Up_Status__c': ['Pending', 'Completed', 'Overdue', 'Not Required'],
    'Customer_Feedback__c': ['Junk', 'Positive', 'Negative', 'Neutral', 'Interested', 'Not Interested'],
    'Transfer_Status__c': ['Pending', 'Transferred', 'Rejected']
}

TASK_FIELD_TYPES = {
    'Id': 'string', 'Lead_Id__c': 'string', 'Opp_Lead_Id__c': 'string', 'Transfer_Status__c': 'category',
    'Customer_Feedback__c': 'category', 'Sales_Team_Feedback__c': 'string', 'Status': 'category',
    'Follow_Up_Status__c': 'category', 'Subject': 'string', 'OwnerId': 'string', 'CreatedById': 'string'
}

FIELD_DISPLAY_NAMES = {
    'Id': 'ID', 'Name': 'Name', 'Status': 'Status', 'LeadSource': 'Lead Source', 'CreatedDate': 'Created Date',
    'Customer_Feedback__c': 'Customer Feedback', 'Project_Category__c': 'Project Category', 'Property_Type__c': 'Property Type',
    'Property_Size__c': 'Property Size', 'Rating': 'Rating', 'Disqualification_Reason__c': 'Disqualification Reason',
    'Type': 'Type', 'Feedback__c': 'Feedback', 'Appointment_Status__c': 'Appointment Status', 'StageName': 'Stage Name',
    'Amount': 'Amount', 'CloseDate': 'Close Date', 'Opportunity_Type__c': 'Opportunity Type', 'Sales_Order_Number__c': 'Sales Order Number',
    'OwnerId': 'Owner ID', 'Phone__c': 'Phone', 'Service_Request_Number__c': 'Service Request Number', 'Subject': 'Subject',
    'StartDateTime': 'Start DateTime', 'EndDateTime': 'End DateTime', 'Transfer_Status__c': 'Transfer Status',
    'Follow_Up_Status__c': 'Follow-Up Status'
}

FIELD_VALUES = {
    'Status': ['Qualified', 'Unqualified', 'Open', 'Converted', 'Disqualified'],
    'LeadSource': ['Website', 'Facebook', 'Google Ads', 'Referral', 'Email Campaign', 'Event', 'Phone Inquiry'],
    'Property_Size__c': ['1BHK', '2BHK', '3BHK', '4BHK', 'Villa'],
    'Property_Type__c': ['Apartment', 'Independent House', 'Villa', 'Plot', 'Commercial'],
    'Purpose_of_Purchase__c': ['Investment', 'Self-Use', 'Rental', 'Resale'],
    'Budget_Range__c': ['<1Cr', '1-2Cr', '1.5Cr', '2-3Cr', '3-5Cr', '>5Cr'],
    'Contact_on_Whatsapp__c': ['Yes', 'No'], 'Lead_Converted__c': ['Yes', 'No'],
    'Same_As_Permanent_Address__c': ['Yes', 'No'], 'Is_Appointment_Booked__c': ['Yes', 'No'],
    'Customer_Interested__c': ['Yes', 'No'], 'Customer_feedback_is_junk__c': ['Yes', 'No'],
    'Customer_Feedback__c': ['Junk', 'Positive', 'Negative', 'Neutral', 'Interested', 'Not Interested']
}

FIELD_TYPES = {
    'CreatedDate': 'datetime', 'Follow_Up_Date_Time__c': 'datetime', 'Disqualification_Date__c': 'date',
    'Disqualified_Date_Time__c': 'datetime', 'Preferred_Date_of_Visit__c': 'date', 'Preferred_Time_of_Visit__c': 'time',
    'Phone__c': 'string', 'Mobile__c': 'string', 'Email__c': 'email', 'IP_Address__c': 'string', 'Budget_Range__c': 'category',
    'Property_Size__c': 'category', 'Status': 'category', 'LeadSource': 'category', 'Max_Price__c': 'float',
    'Min_Price__c': 'float', 'Customer_Feedback__c': 'category'
}

col_display_name = {
    "Name": "User", "Department": "Department", "Meeting_Done_Count": "Completed Meetings"
}

# Field Selection Functions
def get_minimal_lead_fields():
    return ['Id', 'Lead_Id__c', 'Customer_Feedback__c', 'Junk_Reason__c', 'City__c', 'LeadSource', 'Lead_Source_Sub_Category__c',
            'Project__c', 'Project_Category__c', 'Budget_Range__c', 'Property_Size__c', 'Property_Type__c', 'Rating',
            'Status', 'Contact_Medium__c', 'OwnerId', 'CreatedDate', 'Open_Lead_reasons__c', 'Transfer_Status__c',
            'Disqualification_Reason__c', 'Disqualified_Date_Time__c', 'Lead_Converted__c', 'Disqualification_Date__c',
            'Is_Appointment_Booked__c']

def get_standard_lead_fields():
    return get_minimal_lead_fields()

def get_extended_lead_fields():
    return get_minimal_lead_fields()

def get_safe_user_fields():
    return ['Id', 'Name', 'FirstName', 'LastName', 'Email', 'Department', 'Username', 'UserRoleId', 'CreatedDate']

def get_minimal_user_fields():
    return get_safe_user_fields()

def get_standard_user_fields():
    return get_safe_user_fields()

def get_extended_user_fields():
    return get_safe_user_fields()

def get_minimal_case_fields():
    return ['Id', 'Service_Request_Number__c', 'Type', 'Subject', 'CreatedDate', 'Origin', 'Feedback__c', 'Corporate_Closure_Remark__c']

def get_standard_case_fields():
    return ['Id', 'Service_Request_Number__c', 'Type', 'Subject', 'Origin', 'CreatedDate', 'Corporate_Closure_by__c',
            'Corporate_Closure_date__c', 'Description', 'Feedback__c']

def get_extended_case_fields():
    return ['Id', 'Action_Taken__c', 'Back_Office_Date__c', 'Back_Office_Remarks__c', 'Corporate_Closure_Remark__c',
            'Corporate_Closure_by__c', 'Corporate_Closure_date__c', 'Description', 'Feedback__c', 'First_Assigned_By__c',
            'First_Assigned_To__c', 'First_Assigned_on_Date_and_Time__c', 'Opportunity__c', 'Origin', 'Re_Assigned_By1__c',
            'Re_Assigned_On_Date_And_Time__c', 'Re_Assigned_To__c', 'Re_Open_Date_and_time__c', 'Re_Open_by__c',
            'Resolved_By__c', 'Service_Request_Number__c', 'Service_Request_Type__c', 'Service_Sub_catogery__c', 'Subject',
            'Type', 'CreatedDate']

def get_minimal_event_fields():
    return ['Id', 'AccountId', 'Appointment_Status__c', 'CreatedDate', 'OwnerId']

def get_standard_event_fields():
    return ['Id', 'AccountId', 'Appointment_Status__c', 'CreatedById', 'CreatedDate', 'OwnerId']

def get_extended_event_fields():
    return ['Id', 'AccountId', 'Appointment_Status__c', 'CreatedById', 'CreatedDate', 'WhoId', 'OwnerId']

def get_minimal_opportunity_fields():
    return ['Id', 'Lead_Id__c', 'Project__c', 'Project_Category__c', 'CreatedById', 'OwnerId', 'CreatedDate', 'LeadSource',
            'Lead_Source_Sub_Category__c', 'Lead_Rating_By_Sales__c', 'Range_Budget__c', 'Property_Type__c', 'Property_Size__c',
            'Sales_Team_Feedback__c', 'Sales_Open_Reason__c', 'Disqualification_Reason__c', 'Disqualified_Date__c', 'StageName',
            'SAP_Customer_code__c', 'Registration_Number__c', 'Sales_Order_Number__c', 'Sales_Order_Date__c', 'Age__c',
            'SAP_City__c', 'Country_SAP__c', 'State__c']

def get_standard_opportunity_fields():
    return get_minimal_opportunity_fields()

def get_extended_opportunity_fields():
    return get_minimal_opportunity_fields()

def get_minimal_task_fields():
    return ['Id', 'Lead_Id__c', 'Opp_Lead_Id__c', 'Transfer_Status__c', 'Customer_Feedback__c', 'Status', 'Subject', 'OwnerId', 'CreatedById']

def get_standard_task_fields():
    return ['Id', 'Lead_Id__c', 'Opp_Lead_Id__c', 'Transfer_Status__c', 'Customer_Feedback__c', 'Sales_Team_Feedback__c',
            'Status', 'Follow_Up_Status__c', 'Subject', 'OwnerId', 'CreatedById']

def get_extended_task_fields():
    return get_standard_task_fields()


# === Salesforce Utilities Section =======================

async def get_access_token():
    if not all([client_id, client_secret, username, password, login_url]):
        raise ValueError("Missing required Salesforce credentials in environment variables")
    payload = {
        'grant_type': 'password',
        'client_id': client_id,
        'client_secret': client_secret,
        'username': username,
        'password': password
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(login_url, data=payload, timeout=30) as res:
                res.raise_for_status()
                token_data = await res.json()
                return token_data['access_token']
    except Exception as e:
        logger.error(f"Failed to authenticate with Salesforce: {e}")
        raise

async def test_fields_incrementally(session, access_token, base_url, field_sets, object_type="Lead"):
    headers = {'Authorization': f'Bearer {access_token}'}
    field_sets_ordered = {'extended': field_sets['extended'], 'standard': field_sets['standard'], 'minimal': field_sets['minimal']}
    for field_set_name, fields in field_sets_ordered.items():
        test_query = f"SELECT {', '.join(fields)} FROM {object_type} LIMIT 1"
        test_url = base_url + quote(test_query)
        try:
            async with session.get(test_url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    logger.info(f"✅ {object_type} {field_set_name} fields work fine")
                    return fields, field_set_name
                else:
                    text = await response.text()
                    logger.warning(f"❌ {object_type} {field_set_name} failed: {response.status} - {text[:200]}")
        except Exception as e:
            logger.warning(f"❌ {object_type} {field_set_name} error: {e}")
    return None, None

async def debug_individual_fields(session, access_token, base_url, fields_to_test, object_type="Lead"):
    headers = {'Authorization': f'Bearer {access_token}'}
    working_fields = ['Id']
    problematic_fields = []
    for field in fields_to_test:
        if field == 'Id':
            continue
        test_query = f"SELECT Id, {field} FROM {object_type} LIMIT 1"
        test_url = base_url + quote(test_query)
        try:
            async with session.get(test_url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    working_fields.append(field)
                    logger.info(f"✅ {object_type} {field} - OK")
                else:
                    text = await response.text()
                    problematic_fields.append(field)
                    logger.warning(f"❌ {object_type} {field} - FAILED: {text[:100]}")
        except Exception as e:
            problematic_fields.append(field)
            logger.error(f"❌ {object_type} {field} - ERROR: {e}")
    logger.info(f"Working {object_type} fields: {working_fields}")
    logger.info(f"Problematic {object_type} fields: {problematic_fields}")
    return working_fields

def make_arrow_compatible(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype(str).replace('nan', None)
        elif df_copy[col].dtype.name.startswith('datetime'):
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce', utc=True)
    return df_copy

async def fetch_all_pages(session, start_url, headers):
    all_records = []
    next_url = start_url
    while next_url:
        try:
            async with session.get(next_url, headers=headers, timeout=60) as response:
                if response.status == 200:
                    data = await response.json()
                    records = data.get('records', [])
                    all_records.extend(records)
                    next_url = "https://waveinfratech.my.salesforce.com" + data['nextRecordsUrl'] if 'nextRecordsUrl' in data else None
                else:
                    text = await response.text()
                    logger.error(f"Failed to fetch page: {response.status} - {text}")
                    break
        except Exception as e:
            logger.error(f"Error during pagination: {str(e)}")
            break
    return all_records

async def load_object_data(session, access_token, base_url, field_sets, object_type, date_filter=True):
    headers = {'Authorization': f'Bearer {access_token}'}
    
    # Field selection
    field_sets_dict = {
        'minimal': field_sets['minimal'],
        'standard': field_sets['standard'],
        'extended': field_sets['extended']
    }
    fields, field_set_used = await test_fields_incrementally(session, access_token, base_url, field_sets_dict, object_type)
    
    if not fields:
        logger.warning(f"All {object_type} field sets failed, testing individual fields...")
        fields = await debug_individual_fields(session, access_token, base_url, field_sets['extended'], object_type)
    
    if not fields or len(fields) <= 1:
        logger.error(f"Could not find any working {object_type} fields")
        return pd.DataFrame()

    # Build query
    start_date = "2021-04-01T00:00:00Z"
    end_date = "2025-03-31T23:59:59Z"
    date_clause = f" WHERE CreatedDate >= {start_date} AND CreatedDate <= {end_date}" if date_filter else ""
    query = f"SELECT {', '.join(fields)} FROM {object_type}{date_clause} ORDER BY CreatedDate DESC"
    initial_url = base_url + quote(query)
    logger.info(f"Executing {object_type} query with {len(fields)} fields: {field_set_used or 'custom'}")

    # Fetch data
    try:
        all_records = await fetch_all_pages(session, initial_url, headers)
        if not all_records:
            logger.warning(f"No {object_type} records found")
            return pd.DataFrame()
        
        clean_records = [{k: v for k, v in record.items() if k != 'attributes'} for record in all_records]
        df = pd.DataFrame(clean_records)
        df = make_arrow_compatible(df)
        logger.info(f"Successfully loaded {len(df)} {object_type}s")
        
        # Log sample data
        if 'CreatedDate' in df.columns:
            df['CreatedDate'] = pd.to_datetime(df['CreatedDate'], utc=True, errors='coerce')
            invalid_dates = df['CreatedDate'].isna().sum()
            logger.info(f"Invalid CreatedDate values: {invalid_dates}")
            if invalid_dates > 0:
                logger.warning(f"Found {invalid_dates} {object_type}s with invalid CreatedDate values")
        
        return df
    except Exception as e:
        logger.error(f"Error loading {object_type} data: {str(e)}")
        return pd.DataFrame()

async def load_users(session, access_token):
    headers = {'Authorization': f'Bearer {access_token}'}
    try:
        user_fields = get_safe_user_fields()
        user_query = f"SELECT {', '.join(user_fields)} FROM User WHERE IsActive = true LIMIT 200"
        user_url = SF_USERS_URL + quote(user_query)
        
        async with session.get(user_url, headers=headers, timeout=30) as response:
            if response.status == 200:
                users_json = await response.json()
                users_records = users_json.get('records', [])
                clean_records = [{k: v for k, v in record.items() if k != 'attributes'} for record in users_records]
                df = pd.DataFrame(clean_records)
                df = make_arrow_compatible(df)
                logger.info(f"Successfully loaded {len(df)} users")
                return df
            else:
                text = await response.text()
                logger.warning(f"User query failed: {response.status} - {text}")
                return pd.DataFrame()
    except Exception as e:
        logger.warning(f"User query error: {e}")
        return pd.DataFrame()

async def load_salesforce_data_async():
    try:
        access_token = await get_access_token()
        async with aiohttp.ClientSession() as session:
            headers = {'Authorization': f'Bearer {access_token}'}
            
            # Prepare tasks for concurrent execution
            tasks = [
                load_object_data(
                    session, access_token, SF_LEADS_URL,
                    {'minimal': get_minimal_lead_fields(), 
                     'standard': get_standard_lead_fields(), 
                     'extended': get_extended_lead_fields()},
                    "Lead"
                ),
                load_object_data(
                    session, access_token, SF_USERS_URL,
                    {'minimal': get_minimal_user_fields(), 
                     'standard': get_standard_user_fields(), 
                     'extended': get_extended_user_fields()},
                    "User",
                    date_filter=False  # Remove date filter for User data
                ),
                load_object_data(
                    session, access_token, SF_CASES_URL,
                    {'minimal': get_minimal_case_fields(), 
                     'standard': get_standard_case_fields(), 
                     'extended': get_extended_case_fields()},
                    "Case"
                ),
                load_object_data(
                    session, access_token, SF_EVENTS_URL,
                    {'minimal': get_minimal_event_fields(), 
                     'standard': get_standard_event_fields(), 
                     'extended': get_extended_event_fields()},
                    "Event"
                ),
                load_object_data(
                    session, access_token, SF_OPPORTUNITIES_URL,
                    {'minimal': get_minimal_opportunity_fields(), 
                     'standard': get_standard_opportunity_fields(), 
                     'extended': get_extended_opportunity_fields()},
                    "Opportunity"
                ),
                load_object_data(
                    session, access_token, SF_TASKS_URL,
                    {'minimal': get_minimal_task_fields(), 
                     'standard': get_standard_task_fields(), 
                     'extended': get_extended_task_fields()},
                    "Task"
                )
            ]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            
            # Unpack results
            lead_df, user_df, case_df, event_df, opportunity_df, task_df = results
            return lead_df, user_df, case_df, event_df, opportunity_df, task_df, None
            
    except Exception as e:
        error_msg = f"Error loading Salesforce data: {str(e)}"
        logger.error(error_msg)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), error_msg


def load_salesforce_data():
    """Synchronous wrapper for async data loading"""
    try:
        # Get the current event loop or create one if none exists
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(load_salesforce_data_async())
    except Exception as e:
        logger.error(f"Error loading Salesforce data: {str(e)}")
        raise

# === WatsonX Utilities Section =======

def validate_watsonx_config():
    missing_configs = []
    if not WATSONX_API_KEY:
        missing_configs.append("WATSONX_API_KEY")
    if not WATSONX_PROJECT_ID:
        missing_configs.append("WATSONX_PROJECT_ID")
    if missing_configs:
        error_msg = f"Missing WatsonX configuration: {', '.join(missing_configs)}"
        logger.error(error_msg)
        return False, error_msg
    if len(WATSONX_API_KEY.strip()) < 10:
        return False, "WATSONX_API_KEY appears to be invalid (too short)"
    if len(WATSONX_PROJECT_ID.strip()) < 10:
        return False, "WATSONX_PROJECT_ID appears to be invalid (too short)"
    return True, "Configuration valid"

def get_watsonx_token():
    is_valid, validation_msg = validate_watsonx_config()
    if not is_valid:
        raise ValueError(f"Configuration error: {validation_msg}")
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": WATSONX_API_KEY.strip()}
    logger.info("Requesting IBM Cloud IAM token...")
    try:
        response = requests.post(IBM_CLOUD_IAM_URL, headers=headers, data=data, timeout=90)
        logger.info(f"IAM Token Response Status: {response.status_code}")
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data.get("access_token")
            if not access_token:
                raise ValueError("No access_token in response")
            logger.info("Successfully obtained IAM token")
            return access_token
        else:
            error_details = {
                "status_code": response.status_code,
                "response_text": response.text[:1000],
                "headers": dict(response.headers),
                "request_body": data
            }
            logger.error(f"IAM Token request failed: {error_details}")
            raise requests.exceptions.HTTPError(f"IAM API Error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"IAM Token request exception: {str(e)}")
        raise

def create_data_context(leads_df, users_df, cases_df, events_df, opportunities_df, task_df):
    context = {
        "data_summary": {
            "total_leads": len(leads_df),
            "total_users": len(users_df),
            "total_cases": len(cases_df),
            "total_events": len(events_df),
            "total_opportunities": len(opportunities_df),
            "total_tasks": len(task_df),
            "available_lead_fields": list(leads_df.columns) if not leads_df.empty else [],
            "available_user_fields": list(users_df.columns) if not users_df.empty else [],
            "available_case_fields": list(cases_df.columns) if not cases_df.empty else [],
            "available_event_fields": list(events_df.columns) if not leads_df.empty else [],
            "available_opportunity_fields": list(opportunities_df.columns) if not opportunities_df.empty else [],
            "available_task_fields": list(task_df.columns) if not task_df.empty else []
        }
    }
    if not leads_df.empty:
        context["lead_field_info"] = {}
        for col in leads_df.columns:
            sample_values = leads_df[col].dropna().unique()[:5].tolist()
            context["lead_field_info"][col] = {
                "sample_values": [str(v) for v in sample_values],
                "null_count": leads_df[col].isnull().sum(),
                "data_type": str(leads_df[col].dtype)
            }
    if not cases_df.empty:
        context["case_field_info"] = {}
        for col in cases_df.columns:
            sample_values = cases_df[col].dropna().unique()[:5].tolist()
            context["case_field_info"][col] = {
                "sample_values": [str(v) for v in sample_values],
                "null_count": cases_df[col].isnull().sum(),
                "data_type": str(cases_df[col].dtype)
            }
    if not events_df.empty:
        context["event_field_info"] = {}
        for col in events_df.columns:
            sample_values = events_df[col].dropna().unique()[:5].tolist()
            context["event_field_info"][col] = {
                "sample_values": [str(v) for v in sample_values],
                "null_count": events_df[col].isnull().sum(),
                "data_type": str(events_df[col].dtype)
            }
    if not opportunities_df.empty:
        context["opportunity_field_info"] = {}
        for col in opportunities_df.columns:
            sample_values = opportunities_df[col].dropna().unique()[:5].tolist()
            context["opportunity_field_info"][col] = {
                "sample_values": [str(v) for v in sample_values],
                "null_count": opportunities_df[col].isnull().sum(),
                "data_type": str(opportunities_df[col].dtype)
            }
    if not task_df.empty:
        context["task_field_info"] = {}
        for col in task_df.columns:
            sample_value = task_df[col].dropna().unique()[:5].tolist()
            context["task_field_info"][col] = {
                "sample_values": [str(v) for v in sample_values],
                "null_count": task_df[col].isnull().sum(),
                "data_type": str(task_df[col].dtype)
            }
    return context

def query_watsonx_ai(user_question, data_context, leads_df=None, cases_df=None, events_df=None, users_df=None, opportunities_df=None, task_df=None):
    try:
        is_valid, validation_msg = validate_watsonx_config()
        if not is_valid:
            return {"analysis_type": "error", "explanation": f"Configuration error: {validation_msg}"}

        logger.info("Getting WatsonX access token...")
        token = get_watsonx_token()

        sample_lead_fields = ', '.join(data_context['data_summary']['available_lead_fields'])
        sample_user_fields = ', '.join(data_context['data_summary']['available_user_fields'])
        sample_case_fields = ', '.join(data_context['data_summary']['available_case_fields'])
        sample_event_fields = ', '.join(data_context['data_summary']['available_event_fields'])
        sample_opportunity_fields = ', '.join(data_context['data_summary']['available_opportunity_fields'])
        sample_task_fields = ', '.join(data_context['data_summary']['available_task_fields'])
        
        date_filters = {}
        question_lower = user_question.lower()
        current_date = pd.to_datetime(datetime.datetime.now(datetime.timezone.utc))
        
        quarter_patterns = [
            (r'\b(?:q1|quarter\s*1|first\s*quarter)(?:\s+(?:of\s+)?(\d{4}))?\b', 1),
            (r'\b(?:q2|quarter\s*2|second\s*quarter)(?:\s+(?:of\s+)?(\d{4}))?\b', 2),
            (r'\b(?:q3|quarter\s*3|third\s*quarter)(?:\s+(?:of\s+)?(\d{4}))?\b', 3),
            (r'\b(?:q4|quarter\s*4|fourth\s*quarter)(?:\s+(?:of\s+)?(\d{4}))?\b', 4),
        ]
        selected_quarter = None
        for pattern, q_num in quarter_patterns:
            match = re.search(pattern, question_lower)
            if match:
                year_str = match.group(1)
                if year_str:
                    fy_start_year = int(year_str)
                else:
                    now = datetime.datetime.utcnow()
                    fy_start_year = now.year if now.month >= 4 else now.year - 1

                if q_num == 1:
                    start_date = pd.Timestamp(f"{fy_start_year}-04-01T00:00:00Z", tz="UTC")
                    end_date = pd.Timestamp(f"{fy_start_year}-06-30T23:59:59Z", tz="UTC")
                elif q_num == 2:
                    start_date = pd.Timestamp(f"{fy_start_year}-07-01T00:00:00Z", tz="UTC")
                    end_date = pd.Timestamp(f"{fy_start_year}-09-30T23:59:59Z", tz="UTC")
                elif q_num == 3:
                    start_date = pd.Timestamp(f"{fy_start_year}-10-01T00:00:00Z", tz="UTC")
                    end_date = pd.Timestamp(f"{fy_start_year}-12-31T23:59:59Z", tz="UTC")
                elif q_num == 4:
                    start_date = pd.Timestamp(f"{fy_start_year + 1}-01-01T00:00:00Z", tz="UTC")
                    end_date = pd.Timestamp(f"{fy_start_year + 1}-03-31T23:59:59Z", tz="UTC")

                # Apply bounding logic
                start_date = max(start_date, pd.Timestamp("2021-04-01T00:00:00Z", tz="UTC"))
                end_date = min(end_date, pd.Timestamp("2025-07-31T23:59:59Z", tz="UTC"))

                date_filters["CreatedDate"] = {
                    "$gte": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "$lte": end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                }

                selected_quarter = f"Q{q_num} {fy_start_year}-{str(fy_start_year+1)[-2:]}"
                
                logger.info(f"Detected fiscal quarter: {selected_quarter}")
                break
        

        # Detect date range (e.g., "between 4 April 2024 and 10 May 2024")
        date_range_pattern_1 = r'between\s+(\d{1,2})(?:th|rd|st|nd)?\s*(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\s*(\d{4})\s+and\s+(\d{1,2})(?:th|rd|st|nd)?\s*(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\s*(\d{4})'
        date_range_pattern_2 = r'\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\s*(\d{4})\s*to\s*(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\s*(\d{4})'

        month_mapping = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
            'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }

        # Check full date range with 'between' keyword
        date_range_match_1 = re.search(date_range_pattern_1, question_lower, re.IGNORECASE)
        if date_range_match_1:
            start_day, start_month_str, start_year, end_day, end_month_str, end_year = date_range_match_1.groups()
            start_month = month_mapping.get(start_month_str.lower())
            end_month = month_mapping.get(end_month_str.lower())
            if start_month and end_month:
                try:
                    start_date = pd.to_datetime(f"{start_year}-{start_month}-{start_day}T00:00:00Z", utc=True)
                    end_date = pd.to_datetime(f"{end_year}-{end_month}-{end_day}T23:59:59Z", utc=True)
                    start_date = max(start_date, pd.to_datetime("2021-04-01T00:00:00Z", utc=True))
                    end_date = min(end_date, pd.to_datetime("2025-07-31T23:59:59Z", utc=True))
                    date_filters["CreatedDate"] = {
                        "$gte": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "$lte": end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                except ValueError as e:
                    logger.warning(f"Invalid date range parsed: {e}")
                    date_filters = {}

        # Check month-year to month-year pattern
        elif re.search(date_range_pattern_2, question_lower, re.IGNORECASE):
            match = re.search(date_range_pattern_2, question_lower, re.IGNORECASE)
            start_month_str, start_year, end_month_str, end_year = match.groups()
            start_month = month_mapping.get(start_month_str.lower())
            end_month = month_mapping.get(end_month_str.lower())
            if start_month and end_month:
                try:
                    start_date = pd.to_datetime(f"{start_year}-{start_month}-01T00:00:00Z", utc=True)
                    end_date = pd.to_datetime(f"{end_year}-{end_month}-01", utc=True) + pd.offsets.MonthEnd(1)
                    end_date = pd.to_datetime(end_date.strftime("%Y-%m-%dT23:59:59Z"), utc=True)
                    start_date = max(start_date, pd.to_datetime("2021-04-01T00:00:00Z", utc=True))
                    end_date = min(end_date, pd.to_datetime("2025-07-31T23:59:59Z", utc=True))
                    date_filters["CreatedDate"] = {
                        "$gte": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "$lte": end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                except ValueError as e:
                    logger.warning(f"Invalid month-year range parsed: {e}")
                    date_filters = {}

        # Detect single date (e.g., "4 January 2024")
        date_pattern = r'\b(\d{1,2})(?:th|rd|st|nd)?\s*(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\s*(\d{4})\b'
        date_match = re.search(date_pattern, question_lower, re.IGNORECASE)
        if date_match and not date_filters:
            day = int(date_match.group(1))
            month_str = date_match.group(2).lower()
            year = int(date_match.group(3))
            month = month_mapping.get(month_str)
            if month:
                try:
                    specific_date = pd.to_datetime(f"{year}-{month}-{day}T00:00:00Z", utc=True)
                    specific_date = max(specific_date, pd.to_datetime("2021-04-01T00:00:00Z", utc=True))
                    specific_date = min(specific_date, pd.to_datetime("2025-07-31T23:59:59Z", utc=True))
                    date_filters["CreatedDate"] = {
                        "$gte": specific_date.strftime("%Y-%m-%dT00:00:00Z"),
                        "$lte": specific_date.strftime("%Y-%m-%dT23:59:59Z")
                    }
                except ValueError as e:
                    logger.warning(f"Invalid date parsed: {e}")
                    date_filters = {}

        # Detect Hinglish year (e.g., "2024 ka data")
        hinglish_year_pattern = r'\b(\d{4})\s*ka\s*data\b'
        hinglish_year_match = re.search(hinglish_year_pattern, question_lower, re.IGNORECASE)
        
        if hinglish_year_match and not date_filters:
            year = hinglish_year_match.group(1)
            year_start = pd.to_datetime(f"{year}-01-01T00:00:00Z", utc=True)
            year_end = pd.to_datetime(f"{year}-12-31T23:59:59Z", utc=True)
            year_start = max(year_start, pd.to_datetime("2021-04-01T00:00:00Z", utc=True))
            year_end = min(year_end, pd.to_datetime("2025-07-31T23:59:59Z", utc=True))
            date_filters["CreatedDate"] = {
                "$gte": year_start.strftime("%Y-%m-%dT00:00:00Z"),
                "$lte": year_end.strftime("%Y-%m-%dT23:59:59Z")
            }

        # Detect "today", "last week", "last month"
        current_date = pd.to_datetime(datetime.datetime.now(datetime.timezone.utc))
        if "today" in question_lower and not date_filters:
            date_filters["CreatedDate"] = {
                "$gte": current_date.strftime("%Y-%m-%dT00:00:00Z"),
                "$lte": current_date.strftime("%Y-%m-%dT23:59:59Z")
            }
        elif "last week" in question_lower and not date_filters:
            last_week_end = current_date - pd.Timedelta(days=current_date.weekday() + 1)
            last_week_start = last_week_end - pd.Timedelta(days=6)
            last_week_start = max(last_week_start, pd.to_datetime("2021-04-01T00:00:00Z", utc=True))
            last_week_end = min(last_week_end, pd.to_datetime("2025-07-31T23:59:59Z", utc=True))
            date_filters["CreatedDate"] = {
                "$gte": last_week_start.strftime("%Y-%m-%dT00:00:00Z"),
                "$lte": last_week_end.strftime("%Y-%m-%dT23:59:59Z")
            }
        elif "last month" in question_lower and not date_filters:
            last_month_end = (current_date.replace(day=1) - pd.Timedelta(days=1)).replace(hour=23, minute=59, second=59)
            last_month_start = last_month_end.replace(day=1, hour=0, minute=0, second=0)
            last_month_start = max(last_month_start, pd.to_datetime("2021-04-01T00:00:00Z", utc=True))
            last_month_end = min(last_month_end, pd.to_datetime("2025-07-31T23:59:59Z", utc=True))
            date_filters["CreatedDate"] = {
                "$gte": last_month_start.strftime("%Y-%m-%dT00:00:00Z"),
                "$lte": last_month_end.strftime("%Y-%m-%dT23:59:59Z")
            }
        # Detect dynamic phrases like "last 10 days", "past 2 weeks"
        relative_pattern = r'\b(last|past|previous)\s+(\d+)\s+(day|days|week|weeks|month|months)\b'
        relative_match = re.search(relative_pattern, question_lower)
        if relative_match and not date_filters:
            _, number_str, unit = relative_match.groups()
            try:
                number = int(number_str)
                if unit.startswith('day'):
                    delta = pd.Timedelta(days=number - 1)
                elif unit.startswith('week'):
                    delta = pd.Timedelta(weeks=number) - pd.Timedelta(days=1)
                elif unit.startswith('month'):
                    # Approximate month as 30 days
                    delta = pd.Timedelta(days=number * 30 - 1)
                else:
                    delta = pd.Timedelta(days=0)

                start_date = current_date - delta
                end_date = current_date

                # Apply min/max bounds
                start_date = max(start_date, pd.to_datetime("2021-04-01T00:00:00Z", utc=True))
                end_date = min(end_date, pd.to_datetime("2025-07-31T23:59:59Z", utc=True))

                date_filters["CreatedDate"] = {
                    "$gte": start_date.strftime("%Y-%m-%dT00:00:00Z"),
                    "$lte": end_date.strftime("%Y-%m-%dT23:59:59Z")
                }
            except ValueError as ve:
                logger.warning(f"Error parsing dynamic date range: {ve}")

        # Function to add date filters to response
        def add_date_filters_to_response(response):
            if date_filters:
                response.setdefault("filters", {}).update(date_filters)
                if selected_quarter:
                    response["quarter"] = selected_quarter
                    response["explanation"] += f" (Filtered for {selected_quarter}: {date_filters['CreatedDate']['$gte']} to {date_filters['CreatedDate']['$lte']})"
                else:
                    response["explanation"] += f" (Filtered for date range: {date_filters['CreatedDate']['$gte']} to {date_filters['CreatedDate']['$lte']})"
            return response
        

        

        if any(keyword in question_lower for keyword in ["product wise funnel", "product-wise funnel"]):
            response = {
                "analysis_type": "product_wise_funnel",
                "object_type": "lead",
                "fields": ["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c", "Project_Category__c"],
                "group_by": "Project_Category__c",
                "explanation": "Compute lead conversion funnel metrics (Total Leads, Valid Leads, SOL, Meeting Booked, Disqualified Leads, Open Leads, Total Appointment, Junk %, VL:SOL, SOL:MB, MB:MD, Meeting Done) grouped by Project_Category__c"
            }
            return add_date_filters_to_response(response)

        if any(keyword in question_lower for keyword in ["location wise funnel", "location-wise funnel", "highest lead conversion", "fastest lead conversion"]):
            response = {
                "analysis_type": "location_wise_funnel",
                "object_type": "lead",
                "fields": ["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c", "City__c", "OwnerId"],
                "group_by": "City__c",
                "join": {
                    "table": "opportunities_df",
                    "left_on": "OwnerId",
                    "right_on": "CreatedById",
                    "fields": ["Sales_Order_Number__c"]
                },
                "explanation": "Compute lead conversion funnel metrics (Meeting Booked, Sale Done, MB:SD Ratio) grouped by City__c. Join lead OwnerId with opportunity CreatedById."
            }
            return add_date_filters_to_response(response)

        if any(keyword in question_lower for keyword in ["project wise funnel", "project-wise funnel"]):
            response = {
                "analysis_type": "project_wise_funnel",
                "object_type": "lead",
                "fields": ["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c", "Project__c"],
                "group_by": "Project__c",
                "explanation": "Compute lead conversion funnel metrics (Total Leads, Valid Leads, SOL, Meeting Booked, Disqualified Leads, Open Leads, Total Appointment, Junk %, VL:SOL, SOL:MB, MB:MD, Meeting Done) grouped by Project__c"
            }
            return add_date_filters_to_response(response)

        if any(keyword in question_lower for keyword in ["source wise funnel", "source-wise funnel"]):
            response = {
                "analysis_type": "source_wise_funnel",
                "object_type": "lead",
                "fields": ["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c", "LeadSource"],
                "group_by": "LeadSource",
                "explanation": "Compute lead conversion funnel metrics (Total Leads, Valid Leads, SOL, Meeting Booked, Disqualified Leads, Open Leads, Total Appointment, Junk %, VL:SOL, SOL:MB, MB:MD, Meeting Done) grouped by LeadSource"
            }
            return add_date_filters_to_response(response)

        if any(keyword in question_lower for keyword in ["user wise funnel", "user-wise funnel"]):
            response = {
                "analysis_type": "user_wise_funnel",
                "object_type": "lead",
                "fields": ["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c", "OwnerId"],
                "group_by": "OwnerId",
                "joins": [
                    {
                        "table": "users_df",
                        "left_on": "OwnerId",
                        "right_on": "Id",
                        "fields": ["Name"]
                    },
                    {
                        "table": "users_df",
                        "left_on": "CreatedById",
                        "right_on": "Id",
                        "fields": ["Name"]
                    },
                    {
                        "table": "users_df",
                        "left_on": "CreatedById",
                        "right_on": "Id",
                        "fields": ["Name"]
                    }
                ],
                "explanation": "Compute lead conversion funnel metrics (Total Leads, Valid Leads, SOL, Meeting Booked, Disqualified Leads, Open Leads, Total Appointment, Junk %, VL:SOL, SOL:MB, MB:MD, Meeting Done) grouped by Lead OwnerId, joining events and opportunities CreatedById with user names."
            }
            return add_date_filters_to_response(response)

        if any(keyword in question_lower for keyword in ["user wise follow up", "user-wise follow-up"]):
            response = {
                "analysis_type": "user_wise_follow_up",
                "object_type": "task",
                "fields": ["Subject", "OwnerId"],
                "group_by": ["OwnerId", "Subject"],
                "joins": [
                    {
                        "table": "users_df",
                        "left_on": "OwnerId",
                        "right_on": "Id",
                        "fields": ["Name"]
                    }
                ],
                "explanation": "Compute the count of tasks grouped by Lead OwnerId and Subject, joining the task OwnerId with user names to display results in the format: user | Subject | Count."
            }
            return add_date_filters_to_response(response)

        if any(keyword in question_lower for keyword in ["project wise follow up", "project wise sale follow up", "project-wise follow-up", "project-wise sale-follow-up"]):
            response = {
                "analysis_type": "project_wise_follow_up",
                "object_type": "task",
                "fields": ["Subject", "OwnerId"],
                "group_by": ["Project__c", "Subject"],
                "joins": [
                    {
                        "table": "leads_df",
                        "left_on": "OwnerId",
                        "right_on": "OwnerId",
                        "fields": ["Project__c"]
                    }
                ],
                "explanation": "Compute the count of tasks grouped by Project__c and Subject, joining task OwnerId with lead OwnerId to map projects from leads_df, displaying results in the format: Project | Subject | Count."
            }
            return add_date_filters_to_response(response)

        if any(keyword in question_lower for keyword in ["product wise follow up", "product wise sale follow up", "product-wise follow-up", "product-wise sale-follow-up"]):
            response = {
                "analysis_type": "product_wise_follow_up",
                "object_type": "task",
                "fields": ["Subject", "OwnerId"],
                "group_by": ["Project_Category__c", "Subject"],
                "joins": [
                    {
                        "table": "leads_df",
                        "left_on": "OwnerId",
                        "right_on": "OwnerId",
                        "fields": ["Project_Category__c"]
                    }
                ],
                "explanation": "Compute the count of tasks grouped by Project_Category__c and Subject, joining task OwnerId with lead OwnerId to map product categories from leads_df, displaying results in the format: Product | Subject | Count."
            }
            return add_date_filters_to_response(response)

        if any(keyword in question_lower for keyword in ["source wise follow up", "source wise sale follow up", "source-wise follow-up", "source-wise sale-follow-up"]):
            response = {
                "analysis_type": "source_wise_follow_up",
                "object_type": "task",
                "fields": ["Subject", "OwnerId"],
                "group_by": ["LeadSource", "Subject"],
                "joins": [
                    {
                        "table": "leads_df",
                        "left_on": "OwnerId",
                        "right_on": "OwnerId",
                        "fields": ["LeadSource"]
                    }
                ],
                "explanation": "Compute the count of tasks grouped by LeadSource and Subject, joining task OwnerId with lead OwnerId to map lead sources from leads_df, displaying results in the format: Source | Subject | Count."
            }
            return add_date_filters_to_response(response)

        if any(keyword in question_lower for keyword in ["open lead not follow up", "open leads no follow up", "open lead without follow up", "Which open leads have not been followed up"]):
            response = {
                "analysis_type": "open_lead_not_follow_up",
                "object_type": "task",
                "fields": ["Subject", "OwnerId"],
                "group_by": ["Customer_Feedback__c", "Subject"],
                "joins": [
                    {
                        "table": "leads_df",
                        "left_on": "OwnerId",
                        "right_on": "OwnerId",
                        "fields": ["Customer_Feedback__c"]
                    }
                ],
                "explanation": "Compute the count of tasks grouped by Customer_Feedback__c and Subject, joining task OwnerId with lead OwnerId to map customer feedback from leads_df, displaying results in the format: Customer | Subject | Count."
            }
            return add_date_filters_to_response(response)

        if any(keyword in question_lower for keyword in ["crm team member", "crm-team member", "lead to qualification ratio"]):
            response = {
                "analysis_type": "crm_team_member",
                "object_type": "lead",
                "fields": ["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c", "OwnerId"],
                "group_by": "OwnerId",
                "join": {"table": "users_df", "left_on": "OwnerId", "right_on": "Id", "fields": ["Name"]},
                "explanation": "Compute lead conversion funnel metrics (Total Leads, Valid Leads, SOL, Meeting Booked, Disqualified Leads, Open Leads, Total Appointment, Junk %, VL:SOL, SOL:MB, MB:MD, Meeting Done) grouped by OwnerId, joining with users_df to display user names"
            }
            return add_date_filters_to_response(response)

        if any(keyword in question_lower for keyword in ["sale by user", "user wise sale", "user-wise sale", "sales by user"]):
            response = {
                "analysis_type": "user_sales_summary",
                "object_type": "opportunity",
                "fields": ["OwnerId", "Sales_Order_Number__c"],
                "filters": {"Sales_Order_Number__c": {"$ne": None}},
                "explanation": "Show count of closed-won opportunities grouped by user"
            }
            return add_date_filters_to_response(response)

        if "user wise meeting done" in question_lower:
            response = {
                "analysis_type": "user_meeting_summary",
                "object_type": "event",
                "fields": ["OwnerId", "Appointment_Status__c"],
                "filters": {"Appointment_Status__c": "Completed"},
                "explanation": "Show count of completed meetings grouped by user"
            }
            return add_date_filters_to_response(response)

        if "department wise meeting done" in question_lower:
            response = {
                "analysis_type": "dept_user_meeting_summary",
                "object_type": "event",
                "fields": ["OwnerId", "Appointment_Status__c", "Department"],
                "filters": {"Appointment_Status__c": "Completed"},
                "explanation": "Show count of completed meetings grouped by user and department"
            }
            return add_date_filters_to_response(response)

        if "total meeting done" in question_lower:
            response = {
                "analysis_type": "count",
                "object_type": "event",
                "fields": ["Appointment_Status__c"],
                "filters": {"Appointment_Status__c": "Completed"},
                "explanation": "Show total count of completed meetings"
            }
            return add_date_filters_to_response(response)

        if "disqualification reason" in question_lower or "disqualification reasons" in question_lower:
            response = {
                "analysis_type": "disqualification_summary",
                "object_type": "lead",
                "field": "Disqualification_Reason__c",
                "filters": {},
                "explanation": "Show disqualification reasons with count and percentage"
            }
            return add_date_filters_to_response(response)

        if "junk reason" in question_lower:
            response = {
                "analysis_type": "junk_reason_summary",
                "object_type": "lead",
                "field": "Junk_Reason__c",
                "filters": {},
                "explanation": "Show junk reasons with count and percentage"
            }
            return add_date_filters_to_response(response)

        if any(keyword in question_lower for keyword in ["disqualification"]) and any(pct in question_lower for pct in ["%", "percent", "percentage"]):
            response = {
                "analysis_type": "percentage",
                "object_type": "lead",
                "fields": ["Customer_Feedback__c"],
                "filters": {"Customer_Feedback__c": "Not Interested"},
                "explanation": "Calculate percentage of disqualification leads where Customer_Feedback__c is 'Not Interested'"
            }
            return add_date_filters_to_response(response)

        # Define keyword-to-column mappings
        lead_keyword_mappings = {
            "current lead funnel": "Status",
            "disqualification reasons": "Disqualification_Reason__c",
            "conversion rates": "Status",
            "lead source subcategory": "Lead_Source_Sub_Category__c",
            "(Facebook, Google, Website)": "Lead_Source_Sub_Category__c",
            "customer feedback": "Customer_Feedback__c",
            "interested": "Customer_Feedback__c",
            "not interested": "Customer_Feedback__c",
            "property size": "Property_Size__c",
            "property type": "Property_Type__c",
            "bhk": "Property_Size__c",
            "2bhk": "Property_Size__c",
            "3bhk": "Property_Size__c",
            "residential": "Property_Type__c",
            "commercial": "Property_Type__c",
            "rating": "Property_Type__c",
            "budget range": "Budget_Range__c",
            "frequently requested product": "Project_Category__c",
            "time frame": "Preferred_Date_of_Visit__c",
            "location": "Preferred_Location__c",
            "crm team member": "OwnerId",
            "lead to sale ratio": "Status",
            "time between contact and conversion": "CreatedDate",
            "junk lead": "Customer_Feedback__c",
            "idle lead": "Follow_Up_Date_Time__c",
            "seasonality pattern": "CreatedDate",
            "quality lead": "Rating",
            "time gap": "CreatedDate",
            "missing location": "Preferred_Location__c",
            "product preference": "Project_Category__c",
            "project": "Project__c",
            "project name": "Project__c",
           
            "budget preference": "Budget_Range__c",
            "campaign": "Campaign_Name__c",
            "open lead": "Customer_Feedback__c",
            "not open lead": "Customer_Feedback__c",
            "hot lead": "Rating",
            "cold lead": "Rating",
            "warm lead": "Rating",
            "product": "Project_Category__c",
            "product name": "Project_Category__c",
            "disqualified": "Status",
            "disqualification": "Customer_Feedback__c",
            "unqualified": "Status",
            "qualified": "Status",
            "lead conversion funnel": "Status",
            "funnel analysis": "Status",
            "Junk": "Customer_Feedback__c",
            "aranyam valley": "Project_Category__c",
            "armonia villa": "Project_Category__c",
            "comm booth": "Project_Category__c",
            "commercial plots": "Project_Category__c",
            "dream bazaar": "Project_Category__c",
            "dream homes": "Project_Category__c",
            "eden": "Project_Category__c",
            "eligo": "Project_Category__c",
            "ews": "Project_Category__c",
            "ews_001_(410)": "Project_Category__c",
            "executive floors": "Project_Category__c",
            "fsi": "Project_Category__c",
            "generic": "Project_Category__c",
            "golf range": "Project_Category__c",
            "harmony greens": "Project_Category__c",
            "hssc": "Project_Category__c",
            "hubb": "Project_Category__c",
            "institutional": "Project_Category__c",
            "institutional_we": "Project_Category__c",
            "lig": "Project_Category__c",
            "lig_001_(310)": "Project_Category__c",
            "livork": "Project_Category__c",
            "mayfair park": "Project_Category__c",
            "new plots": "Project_Category__c",
            "none": "Project_Category__c",
            "old plots": "Project_Category__c",
            "plot-res-if": "Project_Category__c",
            "plots-comm": "Project_Category__c",
            "plots-res": "Project_Category__c",
            "prime floors": "Project_Category__c",
            "sco": "Project_Category__c",
            "sco.": "Project_Category__c",
            "swamanorath": "Project_Category__c",
            "trucia": "Project_Category__c",
            "veridia": "Project_Category__c",
            "veridia-3": "Project_Category__c",
            "veridia-4": "Project_Category__c",
            "veridia-5": "Project_Category__c",
            "veridia-6": "Project_Category__c",
            "veridia-7": "Project_Category__c",
            "villas": "Project_Category__c",
            "wave floor": "Project_Category__c",
            "wave floor 85": "Project_Category__c",
            "wave floor 99": "Project_Category__c",
            "wave galleria": "Project_Category__c",
            "wave garden": "Project_Category__c",
            "wave garden gh2-ph-2": "Project_Category__c"
        }

        case_keyword_mappings = {
            "case type": "Type",
            "feedback": "Feedback__c",
            "service request": "Service_Request_Number__c",
            "origin": "Origin",
            "closure remark": "Corporate_Closure_Remark__c"
        }

        event_keyword_mappings = {
            "event status": "Appointment_Status__c",
            "scheduled event": "Appointment_Status__c",
            "cancelled event": "Appointment_Status__c",
            "total appointments": "Appointment_status__c",
            "user wise meeting done": ["OwnerId", "Appointment_Status__c"]
        }

        opportunity_keyword_mappings = {
            "qualified opportunity": "Sales_Team_Feedback__c",
            "disqualified opportunity": "Sales_Team_Feedback__c",
            "amount": "Amount",
            "close date": "CloseDate",
            "opportunity type": "Opportunity_Type__c",
            "new business": "Opportunity_Type__c",
            "renewal": "Opportunity_Type__c",
            "upsell": "Opportunity_Type__c",
            "cross-sell": "Opportunity_Type__c",
            "total sale": "Sales_Order_Number__c",
            "source-wise sale": "LeadSource",
            "source with sale": "LeadSource",
            "lead source subcategory with sale": "Lead_Source_Sub_Category__c",
            "subcategory with sale": "Lead_Source_Sub_Category__c",
            "product-wise sales": "Project_Category__c",
            "products with sales": "Project_Category__c",
            "product sale": "Project_Category__c",
            "project-wise sales": "Project__c",
            "project with sale": "Project__c",
            "project sale": "Project__c",
            "sales by user": "OwnerId",
            "user-wise sale": "OwnerId"
        }

        task_keyword_mappings = {
            "task feedback": "Customer_Feedback__c",
            "sales feedback": "Sales_Team_Feedback__c",
            "transfer status": "Transfer_Status__c",
            "Sales Follow Up": "Subject",
            "Follow Up":  "Subject",
            "Follow_Up_Date_Time__c": "CreatedDate",
            "CreatedDate": "CreatedDate",
            "follow up": "Subject",
            "follow-up": "Subject",
            "sale follow up": "Subject",
            "sales follow up": "Subject",
            "sale-follow up": "Subject",
            "sales-follow up": "Subject"
        }

        system_prompt = f"""
You are an intelligent Salesforce analytics assistant. Your task is to convert user questions into a JSON-based analysis plan for lead, case, event, opportunity, or task data.

Available lead fields: {sample_lead_fields}
Available user fields: {sample_user_fields}
Available case fields: {sample_case_fields}
Available event fields: {sample_event_fields}
Available opportunity fields: {sample_opportunity_fields}
Available task fields: {sample_task_fields}

## Keyword-to-Column Mappings
### Lead Data Mappings:
{json.dumps(lead_keyword_mappings, indent=2)}
### Case Data Mappings:
{json.dumps(case_keyword_mappings, indent=2)}
### Event Data Mappings:
{json.dumps(event_keyword_mappings, indent=2)}
### Opportunity Data Mappings:
{json.dumps(opportunity_keyword_mappings, indent=2)}
### Task Data Mappings:
{json.dumps(task_keyword_mappings, indent=2)}

## Instructions:
- Detect if the question pertains to leads, cases, events, opportunities, or tasks based on keywords like "lead", "case", "event", "opportunity", or "task".
- Use keyword-to-column mappings to select the correct field (e.g., "disqualified opportunity" → `Sales_Team_Feedback__c` for opportunities).
- Apply date filters if provided: {json.dumps(date_filters) if date_filters else "None"}.
- If a quarter is specified, include it in the response: {selected_quarter if selected_quarter else "None"}.
- For all queries, ensure date filters are applied if available.
- For terms like "2BHK", "3BHK", filter `Property_Size__c` (e.g., `Property_Size__c: ["2BHK", "3BHK"]`).
- For "residential" or "commercial", filter `Property_Type__c` (e.g., `Property_Type__c: "Residential"`).
- For project categories (e.g., "ARANYAM VALLEY"), filter `Project_Category__c` (e.g., `Project_Category__c: "ARANYAM VALLEY"`).
- For "interested", filter `Customer_Feedback__c = "Interested"`.
- For "qualified opportunity", filter `Sales_Team_Feedback__c = "Qualified"`.
- For "disqualified opportunity", filter `Sales_Team_Feedback__c = "Disqualified"`.
- For "hot lead", "cold lead", "warm lead", filter `Rating` (e.g., `Rating: "Hot"`).
- For "qualified", filter `Customer_Feedback__c = "Interested"`.
- For "disqualified", "disqualification", or "unqualified", filter `Customer_Feedback__c = "Not Interested"`.
- For "total sale", filter `Sales_Order_Number__c` where it is not null (i.e., `Sales_Order_Number__c: {{"$ne": null}}`) for opportunities to count completed sales.
- For "sale", filter `Sales_Order_Number__c` where it is not null (i.e., `Sales_Order_Number__c: {{"$ne": null}}`) for opportunities to count completed sales.
- For "product-wise sales" or "products with sales", set `analysis_type` to `distribution`, `object_type` to `opportunity`, `field` to `Project_Category__c`, and filter `Sales_Order_Number__c: {{"$ne": null}}`. Group by `Project_Category__c`. Ensure null values in `Sales_Order_Number__c` are excluded before grouping.
- For "project-wise sale", "project with sale", or "project sale", set `analysis_type` to `distribution`, `object_type` to `opportunity`, `field` to `Project__c`, and filter `Sales_Order_Number__c: {{"$ne": null}}`. Group by `Project__c`.
- For "source-wise sale" or "source with sale", set `analysis_type` to `distribution`, `object_type` to `opportunity`, `field` to `LeadSource`, and filter `Sales_Order_Number__c: {{"$ne": null}}`. Group by `LeadSource`.
- For "lead source subcategory with sale" or "subcategory with sale", set `analysis_type` to `distribution`, `object_type` to `opportunity`, `field` to `Lead_Source_Sub_Category__c`, and filter `Sales_Order_Number__c: {{"$ne": null}}`. Group by `Lead_Source_Sub_Category__c`.
- For "open lead", filter `Customer_Feedback__c` in `["Discussion Pending", None]` (i.e., `Customer_Feedback__c: {{"$in": ["Discussion Pending", None]}}`).
- For "not open lead", filter `Customer_Feedback__c` in `["Not Interested", "Junk", "Interested"]` (i.e., `Customer_Feedback__c: {{"$in": ["Not Interested", "Junk", "Interested"]}}`).
- For "lead convert opportunity" or "lead versus opportunity" queries (including "how many", "breakdown", "show me", or "%"), set `analysis_type` to `opportunity_vs_lead` for counts or `opportunity_vs_lead_percentage` for percentages. Use `Customer_Feedback__c = Interested` for opportunities and count all `Id` for leads.
- Data is available from 2024-04-01T00:00:00Z to 2025-07-31T23:59:59Z. Adjust dates outside this range to the nearest valid date.
- For date-specific queries (e.g., "4 January 2024"), filter `CreatedDate` for that date.
- For "today", use current date (UTC).
- For "last week" or "last month", calculate relative to current date (UTC).
- For Hinglish like "2025 ka data", filter `CreatedDate` for that year.
- For "sale by user" or "user-wise sale", set `analysis_type` to `user_sales_summary`, `object_type` to `opportunity`, and join `opportunities_df` with `users_df` on `OwnerId` to `Id`.
- For non-null filters, use `{{"$ne": null}}`.
- For "project wise funnel" or "project-wise funnel", set `analysis_type` to `project_wise_funnel`, `object_type` to `lead`, and group by `Project__c`. Compute funnel metrics.
- For "product wise funnel" or "product-wise funnel", set `analysis_type` to `product_wise_funnel`, `object_type` to `lead`, and group by `Project_Category__c`. Compute funnel metrics.
- For "location wise funnel", "location-wise funnel", "highest lead conversion", or "fastest lead conversion", set `analysis_type` to `location_wise_funnel`, `object_type` to `lead`, and group by `City__c`. Compute funnel metrics.
- For "source wise funnel" or "source-wise funnel", set `analysis_type` to `source_wise_funnel`, `object_type` to `lead`, and group by `LeadSource`. Compute funnel metrics.
- For "user wise funnel" or "user-wise funnel", set `analysis_type` to `user_wise_funnel`, `object_type` to `lead`, and group by `OwnerId`, joining with `users_df`.
- For "crm team member" or "crm-team member", set `analysis_type` to `crm_team_member`, `object_type` to `lead`, and group by `OwnerId`, joining with `users_df`.
- If the user mentions "task status", use the `Status` field for tasks.
- If the user mentions "Total Appointment", use the `Appointment_Status__c` in ["Completed", "Scheduled", "Cancelled", "No show"].
- If the user mentions "completed task", map to `Status` with value "Completed" for tasks.
- If the user mentions "interested", map to `Customer_Feedback__c` with value "Interested" for leads or tasks.
- If the user mentions "not interested", map to `Customer_Feedback__c` with value "Not Interested" for leads or tasks.
- If the user mentions "meeting done", map to `Appointment_Status__c` with value "Completed" for events.
- If the user mentions "meeting booked", map to `Status` with value "Qualified" for leads.
- If the user mentions "user wise meeting done", set `analysis_type` to `user_meeting_summary`, `object_type` to `event`, and join `events_df` with `users_df` on `OwnerId` to `Id`.

## Quarter Detection:
- Detect quarters from keywords:
  - "Q1", "quarter 1", "first quarter" → "Q1 2024-25" (2024-04-01T00:00:00Z to 2024-06-30T23:59:59Z)
  - "Q2", "quarter 2", "second quarter" → "Q2 2024-25" (2024-07-01T00:00:00Z to 2024-09-30T23:59:59Z)
  - "Q3", "quarter 3", "third quarter" → "Q3 2024-25" (2024-10-01T00:00:00Z to 2024-12-31T23:59:59Z)
  - "Q4", "quarter 4", "fourth quarter" → "Q4 2024-25" (2025-01-01T00:00:00Z to 2025-03-31T23:59:59Z)
  - "Q1 2025-26", "quarter 1 2025-26", "first quarter 2025-26" → "Q1 2025-26" (2025-04-01T00:00:00Z to 2025-06-30T23:59:59Z)
- For `quarterly_distribution`, include `quarter` in the response.
- If no quarter is specified for `quarterly_distribution`, default to "Q1 - Q4".

## Analysis Types:
- count: Count records.
- distribution: Frequency of values.
- filter: List records.
- recent: Recent records.
- top: Top values.
- percentage: Percentage of matching records.
- quarterly_distribution: Group by quarters.
- source_wise_funnel: Group by `LeadSource` and `Lead_Source_Sub_Category__c`.
- conversion_funnel: Compute funnel metrics.
- opportunity_vs_lead: Compare count of leads with opportunities.
- opportunity_vs_lead_percentage: Calculate percentage of leads converted to opportunities.
- user_meeting_summary: Count completed meetings per user.
- dept_user_meeting_summary: Count completed meetings per user and department.
- user_sales_summary: Count closed-won opportunities per user.
- product_wise_funnel: Compute funnel metrics grouped by `Project_Category__c`.
- project_wise_funnel: Compute funnel metrics grouped by `Project__c`.
- location_wise_funnel: Compute funnel metrics grouped by `City__c`.
- source_wise_funnel: Compute funnel metrics grouped by `LeadSource`.
- user_wise_funnel: Compute funnel metrics grouped by `OwnerId`.
- crm_team_member: Compute funnel metrics grouped by `OwnerId`.
- task_follow_up_summary: Filter tasks by `Subject` containing "Follow Up" or "Sales Follow Up".

## Lead Conversion Funnel:
For "lead conversion funnel", "funnel analysis", "product wise funnel", "project wise funnel", "source wise funnel", "user wise funnel", "location wise funnel", "crm team member":
- Fields: `["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c"]` (add `Project_Category__c`, `LeadSource`, `Project__c`, `City__c`, or `OwnerId` for grouping).
- Metrics:
  - Total Leads: All leads.
  - Valid Leads: `Customer_Feedback__c != "Junk"`.
  - SOL: `Status = "Qualified"`.
  - Meeting Booked: `Status = "Qualified"` and `Is_Appointment_Booked__c = True`.
  - Disqualified Leads: `Customer_Feedback__c = "Not Interested"`.
  - Open Leads: `Customer_Feedback__c` in `["Discussion Pending", null]`.
  - Total Appointment: `Appointment_Status__c` in `["Completed", "Scheduled", "Cancelled", "No show"]`.
  - Junk %: ((Total Leads - Valid Leads) / Total Leads) * 100.
  - VL:SOL: Valid Leads / SOL.
  - SOL:MB: SOL / Meeting Booked.
  - MB:MD: Meeting Booked / Meeting Done.
  - Meeting Done: Count Events where `Appointment_Status__c = "Completed"`.

- For opportunities:
  - "disqualified opportunity" → Use `Sales_Team_Feedback__c = "Disqualified"`.
  - "qualified opportunity" → Use `Sales_Team_Feedback__c = "Qualified"`.
  - "total sale" → Use `Sales_Order_Number__c: {{"$ne": null}}`.

- For tasks:
  - "completed task" → Use `Status = "Completed"`.
  - "open task" → Use `Status = "Open"`.
  - "not follow-up" → Use `Subject` with regex `Follow Up` (case-insensitive).
  - "interested" → Use `Customer_Feedback__c = "Interested"`.
  - "not interested" → Use `Customer_Feedback__c = "Not Interested"`.
  - "follow up" or "follow-up" → Use `Subject` with regex `Follow Up` (case-insensitive).
  - "sale follow up", "sales follow up", "sale-follow up", or "sales-follow up" → Use `Subject` with regex `Sales Follow Up` (case-insensitive).

## JSON Response Format:
{{
  "analysis_type": "type_name",
  "object_type": "lead" or "case" or "event" or "opportunity" or "task",
  "field": "field_name",
  "fields": ["field_name"],
  "filters": {{"field1": "value1", "field2": {{"$ne": null}}}},
  "group_by": "field_name",
  "join": {{"table": "table_name", "left_on": "left_field", "right_on": "right_field", "fields": ["field_name"]}},
  "quarter": "Q1 2024-25",
  "limit": 10,
  "explanation": "Explain what will be done"
}}

User Question: {user_question}

Respond with valid JSON only.
"""

        ml_url = f"{WATSONX_URL}/ml/v1/text/generation?version=2023-07-07"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        body = {
            "input": system_prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 400,
                "temperature": 0.2,
                "repetition_penalty": 1.1,
                "stop_sequences": ["\n\n"]
            },
            "model_id": WATSONX_MODEL_ID,
            "project_id": WATSONX_PROJECT_ID
        }

        logger.info(f"Querying WatsonX AI with model: {WATSONX_MODEL_ID}")
        response = requests.post(ml_url, headers=headers, json=body, timeout=90)

        if response.status_code != 200:
            error_msg = f"WatsonX AI Error {response.status_code}: {response.text}"
            logger.error(error_msg)
            return {"analysis_type": "error", "message": error_msg}

        result = response.json()
        generated_text = result.get("results", [{}])[0].get("generated_text", "").strip()
        logger.info(f"WatsonX generated response: {generated_text}")

        try:
            generated_text = re.sub(r'```json\n?', '', generated_text)
            generated_text = re.sub(r'\n?```', '', generated_text)
            generated_text = re.sub(r'\b null\b', 'null', generated_text)
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.info(f"Extracted JSON string: {json_str}")
                analysis_plan = json.loads(json_str)

                if "analysis_type" not in analysis_plan:
                    analysis_plan["analysis_type"] = "filter"
                if "explanation" not in analysis_plan:
                    analysis_plan["explanation"] = "Analysis based on user question"
                if "object_type" not in analysis_plan:
                    analysis_plan["object_type"] = "lead"
                    if "lead" in user_question.lower():
                        analysis_plan["object_type"] = "lead"
                    elif "case" in user_question.lower():
                        analysis_plan["object_type"] = "case"
                    elif "event" in user_question.lower():
                        analysis_plan["object_type"] = "event"
                    elif "opportunity" in user_question.lower():
                        analysis_plan["object_type"] = "opportunity"
                    elif "task" in user_question.lower():
                        analysis_plan["object_type"] = "task"

                # Apply centralized date filters
                analysis_plan = add_date_filters_to_response(analysis_plan)

                if "filters" in analysis_plan:
                    for field, condition in analysis_plan["filters"].items():
                        if isinstance(condition, dict) and "$ne" in condition and condition["$ne"] == "null":
                            condition["$ne"] = None
                        elif isinstance(condition, dict):
                            for key, value in condition.items():
                                if value == "null":
                                    condition[key] = None
                        elif condition == "null":
                            analysis_plan["filters"][field] = None

                logger.info(f"Parsed analysis plan: {analysis_plan}")
                return analysis_plan
            else:
                logger.warning("No valid JSON found in WatsonX response")
                return parse_intent_fallback(user_question, generated_text, date_filters, selected_quarter)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            return parse_intent_fallback(user_question, generated_text, date_filters, selected_quarter)

    except Exception as e:
        error_msg = f"WatsonX query failed: {str(e)}"
        logger.error(error_msg)
        return {"analysis_type": "error", "explanation": error_msg}

def parse_intent_fallback(user_question, ai_response, date_filters=None, selected_quarter=None):
    question_lower = user_question.lower()
    filters = {} if not date_filters else date_filters.copy()
    object_type = "lead"
    if "lead" in question_lower:
        object_type = "lead"
    elif "case" in question_lower:
        object_type = "case"
    elif "event" in question_lower:
        object_type = "event"
    elif "opportunity" in question_lower:
        object_type = "opportunity"
    elif "task" in question_lower:
        object_type = "task"

    # Apply additional filters based on query
    if "disqualified opportunity" in question_lower and object_type == "opportunity":
        filters["Sales_Team_Feedback__c"] = "Disqualified"
    if ("total sale" in question_lower or "sale" in question_lower) and object_type == "opportunity":
        filters["Sales_Order_Number__c"] = {"$ne": None}
    if any(keyword in question_lower for keyword in ["product-wise sales", "products with sales", "product sale"]) and object_type == "opportunity":
        analysis_plan = {
            "analysis_type": "distribution",
            "object_type": "opportunity",
            "fields": ["Project_Category__c"],
            "filters": {"Sales_Order_Number__c": {"$ne": None}},
            "group_by": "Project_Category__c",
            "explanation": "Distribution of sales by Project_Category__c, excluding records where Sales_Order_Number__c is null"
        }
    elif ("project-wise sale" in question_lower or "project with sale" in question_lower or "project sale" in question_lower) and object_type == "opportunity":
        analysis_plan = {
            "analysis_type": "distribution",
            "object_type": "opportunity",
            "fields": ["Project__c"],
            "filters": {"Sales_Order_Number__c": {"$ne": None}},
            "explanation": "Distribution of sales by project"
        }
    elif ("source-wise sale" in question_lower or "source with sale" in question_lower) and object_type == "opportunity":
        analysis_plan = {
            "analysis_type": "distribution",
            "object_type": "opportunity",
            "fields": ["LeadSource"],
            "filters": {"Sales_Order_Number__c": {"$ne": None}},
            "explanation": "Distribution of sales by source"
        }
    elif ("lead source subcategory with sale" in question_lower or "subcategory with sale" in question_lower) and object_type == "opportunity":
        analysis_plan = {
            "analysis_type": "distribution",
            "object_type": "opportunity",
            "fields": ["Lead_Source_Sub_Category__c"],
            "filters": {"Sales_Order_Number__c": {"$ne": None}},
            "explanation": "Distribution of sales by lead source subcategory"
        }
    else:
        analysis_plan = {
            "analysis_type": "filter",
            "object_type": object_type,
            "filters": filters,
            "explanation": f"Filtering {object_type} records for: {user_question}"
        }

    # Apply centralized date filters
    def add_date_filters_to_response(response):
        if date_filters:
            response.setdefault("filters", {}).update(date_filters)
            if selected_quarter:
                response["quarter"] = selected_quarter
                response["explanation"] += f" (Filtered for {selected_quarter}: {date_filters['CreatedDate']['$gte']} to {date_filters['CreatedDate']['$lte']})"
            else:
                response["explanation"] += f" (Filtered for date range: {date_filters['CreatedDate']['$gte']} to {date_filters['CreatedDate']['$lte']})"
        return response

    analysis_plan = add_date_filters_to_response(analysis_plan)
    return analysis_plan

# === Analysis Engine Section ==============================
col_display_name = {
        "Name": "User",
        "Department": "Department",
        "Meeting_Done_Count": "Completed Meetings"
        }


def execute_analysis(analysis_plan, leads_df, users_df, cases_df, events_df, opportunities_df, task_df, user_query, user_question=""):
    """
    Execute the analysis based on the provided plan and dataframes.
    """
    try:
        # Extract analysis parameters
        analysis_type = analysis_plan.get("analysis_type", "filter")
        object_type = analysis_plan.get("object_type", "lead")
        fields = analysis_plan.get("fields", [])
        if "field" in analysis_plan and analysis_plan["field"]:
            if analysis_plan["field"] not in fields:
                fields.append(analysis_plan["field"])
        filters = analysis_plan.get("filters", {})
        selected_quarter = analysis_plan.get("quarter", None)

        logger.info(f"Executing analysis for query '{user_question}': {analysis_plan}")

        # Select the appropriate dataframe based on object_type
        if object_type == "lead":
            df = leads_df
        elif object_type == "case":
            df = cases_df
        elif object_type == "event":
            df = events_df
        elif object_type == "opportunity":
            
            df = opportunities_df
            
            df  = df[
                df["Sales_Order_Number__c"].notna() & 
                (df["Sales_Order_Number__c"] != "None") &
                (df["Sales_Order_Number__c"] != "")
            ]
        

        
        elif object_type == "task":
            df = task_df
        else:
            logger.error(f"Unsupported object_type: {object_type}")
            return {"type": "error", "message": f"Unsupported object type: {object_type}"}
        # Add validation step before filtering to ensure data is present
        if object_type == "opportunity" and (df.empty or 'Sales_Team_Feedback__c' not in df.columns):
            logger.error(f"{object_type}_df is empty or missing Sales_Team_Feedback__c: {df.columns}")
            return {"type": "error", "message": f"No {object_type} data or required column missing"}

        if df.empty:
            logger.error(f"No {object_type} data available")
            return {"type": "error", "message": f"No {object_type} data available"}

        # Detect specific query types
        source_keywords = ["source-wise", "lead source"]
        project_keywords = ["project-wise", "project"]
        user_keywords = ["user-wise", "user based", "employee-wise"]
        product_funnel_keywords = ["product wise funnel", "product-wise funnel"]
        
        is_source_related = any(keyword in user_question.lower() for keyword in source_keywords)
        is_project_related = any(keyword in user_question.lower() for keyword in project_keywords)
        is_user_related = any(keyword in user_question.lower() for keyword in user_keywords)
        is_product_funnel = any(keyword in user_question.lower() for keyword in product_funnel_keywords)

        # Validate fields for opportunity_vs_lead analysis
        if analysis_type in ["opportunity_vs_lead", "opportunity_vs_lead_percentage"]:
            required_fields = ["Customer_Feedback__c", "Id"] 
            missing_fields = [f for f in required_fields if f not in df.columns]
            if missing_fields:
                logger.error(f"Missing fields for {analysis_type}: {missing_fields}")
                return {"type": "error", "message": f"Missing fields: {missing_fields}"}

        if analysis_type in ["distribution", "top", "percentage", "quarterly_distribution", "source_wise_lead", "product_wise_lead", "conversion_funnel", "product_wise_funnel"] and not fields:
            fields = list(filters.keys()) if filters else []
            if not fields:
                logger.error(f"No fields specified for {analysis_type} analysis")
                return {"type": "error", "message": f"No fields specified for {analysis_type} analysis"}

        # Detect specific query types
        product_keywords = ["product sale", "product split", "sale"]
        sales_keywords = ["sale", "sales", "project-wise sale", "source-wise sale", "lead source subcategory with sale"]
        
        is_product_related = any(keyword in user_question.lower() for keyword in product_keywords)
        is_sales_related = any(keyword in user_question.lower() for keyword in sales_keywords)

        # Adjust fields for product-related and sales-related queries
        if is_product_related and object_type == "lead":
            logger.info(f"Detected product-related question: '{user_question}'. Using Project_Category__c and Status.")
            required_fields = ["Project_Category__c", "Status"]
            missing_fields = [f for f in required_fields if f not in df.columns]
            if missing_fields:
                logger.error(f"Missing fields for product analysis: {missing_fields}")
                return {"type": "error", "message": f"Missing fields for product analysis: {missing_fields}"}
            if "Project_Category__c" not in fields:
                fields.append("Project_Category__c")
            if "Status" not in fields:
                fields.append("Status")
            if analysis_type not in ["source_wise_lead", "product_wise_lead", "distribution", "quarterly_distribution", "product_wise_funnel"]:
                analysis_type = "distribution"
                analysis_plan["analysis_type"] = "distribution"
            analysis_plan["fields"] = fields

        if is_sales_related and object_type == "opportunity":
            ("testing data check")
            
            if "Sales_Order_Number__c" in df.columns:
                df = df[df["Sales_Order_Number__c"].notna() & (df["Sales_Order_Number__c"] != "None") & (df["Sales_Order_Number__c"] != "")]
                logger.info(f"Filtered Sales_Order_Number__c for sales-related question. Remaining rows: {len(df)}")
            else:
                logger.error("Sales_Order_Number__c column not found in opportunity data")
                return {"type": "error", "message": "Sales_Order_Number__c column not found in opportunity data"}
            # Enhanced filter to exclude null and "None" values
            df  = df[
                df["Sales_Order_Number__c"].notna() & 
                (df["Sales_Order_Number__c"] != "None") &
                (df["Sales_Order_Number__c"] != "")
            ]
           
            logger.info(f"Opportunities after filtering None Sales_Order_Number__c: {len(df)}")
            # Then apply additional product/project filters if specified
            if "Project_Category__c" in fields or any(f in filters for f in ["Project_Category__c", "Project"]):
                if "Project_Category__c" not in fields:
                    fields.append("Project_Category__c")
                analysis_plan["fields"] = fields
            if analysis_type not in ["distribution", "quarterly_distribution", "product_wise_funnel", "product_wise_sale"]:
                # analysis_type = "distribution"
                # analysis_plan["analysis_type"] = "distribution"
                analysis_type = "product_wise_sale"  # Default to product_wise_sale for sales-related queries
                analysis_plan["analysis_type"] = "product_wise_sale"

        
        filtered_df = df.copy()
        
        # Parse CreatedDate if present
        if 'CreatedDate' in filtered_df.columns:
            logger.info(f"Raw CreatedDate sample (first 5):\n{filtered_df['CreatedDate'].head().to_string()}")
            logger.info(f"Raw CreatedDate dtype: {filtered_df['CreatedDate'].dtype}")
            try:
                def parse_date(date_str):
                    if pd.isna(date_str):
                        return pd.NaT
                    try:
                        return pd.to_datetime(date_str, utc=True, errors='coerce')
                    except:
                        pass
                    try:
                        parsed_date = pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S', errors='coerce')
                        if pd.notna(parsed_date):
                            ist = timezone('Asia/Kolkata')
                            parsed_date = ist.localize(parsed_date).astimezone(timezone('UTC'))
                        return parsed_date
                    except:
                        pass
                    try:
                        parsed_date = pd.to_datetime(date_str, format='%m/%d/%Y', errors='coerce')
                        if pd.notna(parsed_date):
                            ist = timezone('Asia/Kolkata')
                            parsed_date = ist.localize(parsed_date).astimezone(timezone('UTC'))
                        return parsed_date
                    except:
                        pass
                    try:
                        parsed_date = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
                        if pd.notna(parsed_date):
                            ist = timezone('Asia/Kolkata')
                            parsed_date = ist.localize(parsed_date).astimezone(timezone('UTC'))
                        return parsed_date
                    except:
                        return pd.NaT

                filtered_df['CreatedDate'] = filtered_df['CreatedDate'].apply(parse_date)
                invalid_dates = filtered_df[filtered_df['CreatedDate'].isna()]
                if not invalid_dates.empty:
                    logger.warning(f"Found {len(invalid_dates)} rows with invalid CreatedDate values:\n{invalid_dates['CreatedDate'].head().to_string()}")
                filtered_df = filtered_df[filtered_df['CreatedDate'].notna()]
                if filtered_df.empty:
                    logger.error("No valid CreatedDate entries after conversion")
                    return {"type": "error", "message": "No valid CreatedDate entries found in the data"}
                min_date = filtered_df['CreatedDate'].min()
                max_date = filtered_df['CreatedDate'].max()
                logger.info(f"Date range in dataset after conversion (UTC): {min_date} to {max_date}")
            except Exception as e:
                logger.error(f"Error while converting CreatedDate: {str(e)}")
                return {"type": "error", "message": f"Error while converting CreatedDate: {str(e)}"}

        # Apply filters
        for field, value in filters.items():
            if field not in filtered_df.columns:
                logger.error(f"Filter field {field} not in columns: {list(df.columns)}")
                return {"type": "error", "message": f"Field {field} not found"}
            if isinstance(value, str):
                if field in ["Status", "Rating", "Customer_Feedback__c", "LeadSource", "Lead_Source_Sub_Category__c", "Appointment_Status__c", "StageName", "Sales_Team_Feedback__c"]:
                    filtered_df = filtered_df[filtered_df[field].str.lower() == value.lower()]
                else:
                    filtered_df = filtered_df[filtered_df[field].str.contains(value, case=False, na=False)]
            elif isinstance(value, list):
                filtered_df = filtered_df[filtered_df[field].isin(value) & filtered_df[field].notna()]
            elif isinstance(value, dict):
                if field in FIELD_TYPES and FIELD_TYPES[field] == 'datetime':
                    if "$gte" in value:
                        gte_value = pd.to_datetime(value["$gte"], utc=True)
                        filtered_df = filtered_df[filtered_df[field] >= gte_value]
                    if "$lte" in value:
                        lte_value = pd.to_datetime(value["$lte"], utc=True)
                        filtered_df = filtered_df[filtered_df[field] <= lte_value]
                elif "$in" in value:
                    filtered_df = filtered_df[filtered_df[field].isin(value["$in"]) & filtered_df[field].notna()]
                elif "$ne" in value:
                    filtered_df = filtered_df[filtered_df[field] != value["$ne"] if value["$ne"] is not None else filtered_df[field].notna()]
                else:
                    logger.error(f"Unsupported dict filter on {field}: {value}")
                    return {"type": "error", "message": f"Unsupported dict filter on {field}"}
            elif isinstance(value, bool):
                filtered_df = filtered_df[filtered_df[field] == value]
            else:
                filtered_df = filtered_df[filtered_df[field] == value]
            logger.info(f"After filter on {field}: {filtered_df.shape}")

        # Define quarters for 2024-25 financial year
        
        def get_fiscal_quarters(start_year: int):
            """Generate fiscal quarter mappings given the starting year."""
            end_year = start_year + 1
            return {
                f"Q1 {start_year}-{str(end_year)[-2:]}": {
                    "start": pd.to_datetime(f"{start_year}-04-01T00:00:00Z", utc=True),
                    "end": pd.to_datetime(f"{start_year}-06-30T23:59:59Z", utc=True),
                },
                f"Q2 {start_year}-{str(end_year)[-2:]}": {
                    "start": pd.to_datetime(f"{start_year}-07-01T00:00:00Z", utc=True),
                    "end": pd.to_datetime(f"{start_year}-09-30T23:59:59Z", utc=True),
                },
                f"Q3 {start_year}-{str(end_year)[-2:]}": {
                    "start": pd.to_datetime(f"{start_year}-10-01T00:00:00Z", utc=True),
                    "end": pd.to_datetime(f"{start_year}-12-31T23:59:59Z", utc=True),
                },
                f"Q4 {start_year}-{str(end_year)[-2:]}": {
                    "start": pd.to_datetime(f"{end_year}-01-01T00:00:00Z", utc=True),
                    "end": pd.to_datetime(f"{end_year}-03-31T23:59:59Z", utc=True),
                }
            }

        # Apply quarter filter if specified
        
        if selected_quarter and 'CreatedDate' in filtered_df.columns:
            # Extract fiscal year from selected_quarter (e.g., "Q1 2025-26")
            match = re.match(r"Q[1-4]\s+(\d{4})-\d{2}", selected_quarter)
            if match:
                fy_start_year = int(match.group(1))
                quarters = get_fiscal_quarters(fy_start_year)
            else:
                logger.error(f"Invalid quarter format: {selected_quarter}")
                return {"type": "error", "message": f"Invalid quarter specified: {selected_quarter}"}

            quarter = quarters.get(selected_quarter)
            if not quarter:
                logger.warning(f"Quarter not found for {selected_quarter}")
                return {"type": "error", "message": f"Quarter not available: {selected_quarter}"}

            filtered_df['CreatedDate'] = pd.to_datetime(filtered_df['CreatedDate'], utc=True).dt.tz_convert('UTC')
            logger.info(f"Filtering for {selected_quarter}: {quarter['start']} to {quarter['end']}")
            logger.info(f"Sample CreatedDate before filter:\n{filtered_df['CreatedDate'].head().to_string()}")

            filtered_df = filtered_df[
                (filtered_df['CreatedDate'] >= quarter["start"]) &
                (filtered_df['CreatedDate'] <= quarter["end"])
            ]

            logger.info(f"Records after applying quarter filter {selected_quarter}: {len(filtered_df)} rows")
            if not filtered_df.empty:
                logger.info(f"Sample CreatedDate after filter:\n{filtered_df['CreatedDate'].head().to_string()}")
            else:
                logger.warning(f"No records found for {selected_quarter}")

        # Final logging and error handling
        logger.info(f"Final filtered {object_type} DataFrame shape: {filtered_df.shape}")
        if filtered_df.empty:
            return {
                "type": "info",
                "message": f"No {object_type} records found matching the criteria for {selected_quarter if selected_quarter else 'the specified period'}"
            }
        # Prepare graph_data for all analysis types
        graph_data = {}
        graph_fields = fields + list(filters.keys())
        valid_graph_fields = [f for f in graph_fields if f in filtered_df.columns]
        for field in valid_graph_fields:
            if filtered_df[field].dtype in ['object', 'bool', 'category']:
                counts = filtered_df[field].dropna().value_counts().to_dict()
                graph_data[field] = {str(k): v for k, v in counts.items()}
                logger.info(f"Graph data for {field}: {graph_data[field]}")

        # Handle different analysis types
       
        if analysis_type == "opportunity_vs_lead":
            if object_type == "lead":
                # Apply filters (including quarter) to get filtered dataset
                filtered_df = df.copy()
                for field, value in filters.items():
                    if field not in filtered_df.columns:
                        return {"type": "error", "message": f"Field {field} not found"}
                    if isinstance(value, str):
                        filtered_df = filtered_df[filtered_df[field] == value]
                    elif isinstance(value, dict):
                        if "$in" in value:
                            filtered_df = filtered_df[filtered_df[field].isin(value["$in"]) & filtered_df[field].notna()]
                        elif "$ne" in value:
                            filtered_df = filtered_df[filtered_df[field] != value["$ne"]]
                    elif isinstance(value, bool):
                        filtered_df = filtered_df[filtered_df[field] == value]
                if selected_quarter and 'CreatedDate' in filtered_df.columns:
                    quarter = quarters.get(selected_quarter)
                    filtered_df['CreatedDate'] = filtered_df['CreatedDate'].dt.tz_convert('UTC')
                    filtered_df = filtered_df[
                        (filtered_df['CreatedDate'] >= quarter["start"]) &
                        (filtered_df['CreatedDate'] <= quarter["end"])
                    ]
                # Calculate opportunities from the filtered dataset
                opportunities = len(filtered_df[filtered_df["Customer_Feedback__c"] == "Interested"])
                logger.info(f"Opportunities count after filter {selected_quarter if selected_quarter else 'all data'}: {opportunities}")
                data = [
                    {"Category": "Opportunities", "Count": opportunities}
                ]
                graph_data["Opportunity vs Lead"] = {
                    "Opportunities": opportunities
                }
                return {
                    "type": "opportunity_vs_lead",
                    "data": data,
                    "graph_data": graph_data,
                    "filtered_data": filtered_df,
                    "selected_quarter": selected_quarter
                }
        
        # New user_sales_summary analysis type
        elif analysis_type == "user_sales_summary" and object_type == "opportunity":
            required_fields_opp = ["OwnerId", "Sales_Order_Number__c"]
            missing_fields_opp = [f for f in required_fields_opp if f not in opportunities_df.columns]
            if missing_fields_opp:
                logger.error(f"Missing fields in opportunities_df for user_sales_summary: {missing_fields_opp}")
                return {"type": "error", "message": f"Missing fields in opportunities_df: {missing_fields_opp}"}

            if users_df.empty or "Id" not in users_df.columns or "Name" not in users_df.columns:
                logger.error("Users DataFrame is missing or lacks required columns (Id, Name)")
                return {"type": "error", "message": "Users data is missing or lacks Id or Name columns"}

            # Filter opportunities by quarter if specified
            filtered_opp = opportunities_df.copy()
            if selected_quarter and 'CreatedDate' in filtered_opp.columns:
                quarter = quarters.get(selected_quarter)
                filtered_opp['CreatedDate'] = pd.to_datetime(filtered_opp['CreatedDate'], utc=True, errors='coerce')
                filtered_opp = filtered_opp[
                    (filtered_opp['CreatedDate'] >= quarter["start"]) &
                    (filtered_opp['CreatedDate'] <= quarter["end"])
                ]

            # Merge with users_df to get user names
            merged_df = filtered_opp.merge(
                users_df[["Id", "Name"]],
                left_on="OwnerId",
                right_on="Id",
                how="left"
            )

            # Group by user name and count sales orders
            sales_counts = merged_df.groupby("Name")["Sales_Order_Number__c"].count().reset_index(name="Sales_Order_Count")
            sales_counts = sales_counts.sort_values(by="Sales_Order_Count", ascending=False)
            total_sales = len(merged_df)

            # Prepare graph data
            graph_data["User_Sales"] = sales_counts.set_index("Name")["Sales_Order_Count"].to_dict()

            return {
                "type": "user_sales_summary",
                "data": sales_counts.to_dict(orient="records"),
                "columns": ["Name", "Sales_Order_Count"],
                "total": total_sales if not sales_counts.empty else 0,
                "graph_data": graph_data,
                "filtered_data": merged_df,
                "selected_quarter": selected_quarter
            }
            
        if selected_quarter:
            start_date = quarters[selected_quarter]["start"]
            end_date = quarters[selected_quarter]["end"]
            if object_type == "event" and "CreatedDate" in events_df.columns:
                events_df = events_df[(pd.to_datetime(events_df["CreatedDate"], utc=True) >= start_date) & (pd.to_datetime(events_df["CreatedDate"], utc=True) <= end_date)]

        if object_type == "event":
            if analysis_type == "user_meeting_summary":
                required_fields_events = ["OwnerId", "Appointment_Status__c"]
                missing_fields_events = [f for f in required_fields_events if f not in events_df.columns]
                if missing_fields_events:
                    logger.error(f"Missing fields in events_df for user_meeting_summary: {missing_fields_events}")
                    return {"type": "error", "message": f"Missing fields in events_df: {missing_fields_events}"}

                if users_df.empty or "Id" not in users_df.columns or "Name" not in users_df.columns:
                    logger.error("Users DataFrame is missing or lacks required columns (Id, Name)")
                    return {"type": "error", "message": "Users data is missing or lacks Id or Name columns"}

            # Filter for completed meetings
                completed_events = events_df[events_df["Appointment_Status__c"].str.lower() == "completed"]

            # Merge with users_df to get only Name, excluding Department
                merged_df = completed_events.merge(
                    users_df[["Id", "Name"]],
                    left_on="OwnerId",
                    right_on="Id",
                    how="left"
                )

            # Group by User Name only
                user_counts = merged_df.groupby("Name").size().reset_index(name="Meeting_Done_Count")
                user_counts = user_counts.sort_values(by="Meeting_Done_Count", ascending=False)
                total_meetings = len(merged_df)

            # Prepare graph data
                graph_data["User_Meeting_Done"] = user_counts.set_index("Name")["Meeting_Done_Count"].to_dict()

                return {
                    "type": "user_meeting_summary",
                    "data": user_counts.to_dict(orient="records"),
                    "columns": ["Name", "Meeting_Done_Count"],  # Explicitly exclude Department
                    "total": total_meetings if not user_counts.empty else 0,
                    "graph_data": graph_data,
                    "filtered_data": merged_df,
                    "selected_quarter": selected_quarter
                }
        
            elif analysis_type == "dept_user_meeting_summary" and object_type == "event":
                    required_fields_events = ["OwnerId", "Appointment_Status__c"]
                    missing_fields_events = [f for f in required_fields_events if f not in events_df.columns]
                    if missing_fields_events:
                        logger.error(f"Missing fields in events_df for dept_user_meeting_summary: {missing_fields_events}")
                        return {"type": "error", "message": f"Missing fields in events_df: {missing_fields_events}"}

                    if users_df.empty or "Id" not in users_df.columns or "Name" not in users_df.columns or "Department" not in users_df.columns:
                        logger.error("Users DataFrame is missing or lacks required columns (Id, Name, Department)")
                        return {"type": "error", "message": "Users data is missing or lacks Id, Name, or Department columns"}

                # Filter for completed meetings
                    completed_events = events_df[events_df["Appointment_Status__c"].str.lower() == "completed"]

                # Merge with users_df to get only Department
                    merged_df = completed_events.merge(
                        users_df[["Id", "Department"]],
                        left_on="OwnerId",
                        right_on="Id",
                        how="left"
                    )

                # Group by Department only, then count
                    dept_counts = merged_df.groupby("Department").size().reset_index(name="Meeting_Done_Count")
                    dept_counts = dept_counts.sort_values(by="Meeting_Done_Count", ascending=False)
                    total_meetings = len(merged_df)

                # Prepare graph data (using only Department as index)
                    graph_data["Dept_Meeting_Done"] = dept_counts.set_index("Department")["Meeting_Done_Count"].to_dict()

                    return {
                        "type": "dept_user_meeting_summary",
                        "data": dept_counts.to_dict(orient="records"),
                        "columns": ["Department", "Meeting_Done_Count"],  # Only Department and count
                        "total": total_meetings if not dept_counts.empty else 0,
                        "graph_data": graph_data,
                        "filtered_data": merged_df,
                        "selected_quarter": selected_quarter
                    }
        
        # Handle opportunity_vs_lead_percentage analysis
        elif analysis_type == "opportunity_vs_lead_percentage":
            if object_type == "lead":
                total_leads = len(filtered_df)
                opportunities = len(filtered_df[filtered_df["Customer_Feedback__c"] == "Interested"])  
                percentage = (opportunities / total_leads * 100) if total_leads > 0 else 0
                graph_data["Opportunity vs Lead"] = {
                    "Opportunities": percentage,
                    "Non-Opportunities": 100 - percentage
                }
                return {
                    "type": "percentage",
                    "value": round(percentage, 1),
                    "label": "Percentage of Leads Marked as Interested",
                    "graph_data": graph_data,
                    "filtered_data": filtered_df,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"Opportunity vs Lead percentage analysis not supported for {object_type}"}

        elif analysis_type == "count":
            return {
                "type": "metric",
                "value": len(filtered_df),
                "label": f"Total {object_type.capitalize()} Count",
                "graph_data": graph_data,
                "filtered_data": filtered_df,
                "selected_quarter": selected_quarter
            }

        elif analysis_type == "disqualification_summary":
            df = leads_df if object_type == "lead" else opportunities_df
            field = analysis_plan.get("field", "Disqualification_Reason__c")
            if df is None or df.empty:
                return {"type": "error", "message": f"No data available for {object_type}"}
            if field not in df.columns:
                return {"type": "error", "message": f"Field {field} not found in {object_type} data"}

            filtered_df = df[df[field].notna() & (df[field] != "") & (df[field].astype(str).str.lower() != "none")]

            # Generate counts and percentages
            disqual_counts = filtered_df[field].value_counts()
            total = disqual_counts.sum()
            summary = [
                {
                    "Disqualification Reason": str(reason),
                    "Count": count,
                    "Percentage": round((count / total) * 100, 2)
                }
                for reason, count in disqual_counts.items()
            ]
            graph_data[field] = {str(k): v for k, v in disqual_counts.items()}
            return {
                "type": "disqualification_summary",
                "data": summary,
                "field": field,
                "total": total,
                "graph_data": graph_data,
                "filtered_data": filtered_df,
                "selected_quarter": selected_quarter
            }

        elif analysis_type == "junk_reason_summary":
            df = leads_df if object_type == "lead" else opportunities_df
            field = analysis_plan.get("field", "Junk_Reason__c")
            if df is None or df.empty:
                return {"type": "error", "message": f"No data available for {object_type}"}
            if field not in df.columns:
                return {"type": "error", "message": f"Field {field} not found in {object_type} data"}
            filtered_df = df[df[field].notna() & (df[field] != "") & (df[field].astype(str).str.lower() != "none")]
            junk_counts = filtered_df[field].value_counts()
            total = junk_counts.sum()
            summary = [
                {
                    "Junk Reason": str(reason),
                    "Count": count,
                    "Percentage": round((count / total) * 100, 2)
                }
                for reason, count in junk_counts.items()
            ]
            graph_data[field] = {str(k): v for k, v in junk_counts.items()}
            return {
                "type": "junk_reason_summary",
                "data": summary,
                "field": field,
                "total": total,
                "graph_data": graph_data,
                "filtered_data": filtered_df,
                "selected_quarter": selected_quarter
            }

        elif analysis_type == "filter":
            selected_columns = [col for col in filtered_df.columns if col in [
                'Id', 'Name', 'Status', 'LeadSource', 'CreatedDate', 'Customer_Feedback__c',
                'Project_Category__c', 'Property_Type__c', "Property_Size__c", 'Rating',
                'Disqualification_Reason__c', 'Type', 'Feedback__c', 'Appointment_Status__c',
                'StageName', 'Amount', 'CloseDate', 'Opportunity_Type__c'
            ]]
            if not selected_columns:
                selected_columns = filtered_df.columns[:5].tolist()
            result_df = filtered_df[selected_columns]
            return {
                "type": "table",
                "data": result_df.to_dict(orient="records"),
                "columns": selected_columns,
                "graph_data": graph_data,
                "count": len(filtered_df),
                "filtered_data": filtered_df,
                "selected_quarter": selected_quarter
            }

        elif analysis_type == "recent":
            if 'CreatedDate' in filtered_df.columns:
                filtered_df['CreatedDate'] = pd.to_datetime(filtered_df['CreatedDate'], utc=True, errors='coerce')
                filtered_df = filtered_df.sort_values('CreatedDate', ascending=False)
                selected_columns = [col for col in filtered_df.columns if col in [
                    'Id', 'Name', 'Status', 'LeadSource', 'CreatedDate', 'Customer_Feedback__c',
                    'Project_Category__c', 'Property_Type__c', "Property_Size__c", 'Rating',
                    'Disqualification_Reason__c', 'Type', 'Feedback__c', 'Appointment_Status__c',
                    'StageName', 'Amount', 'CloseDate', 'Opportunity_Type__c'
                ]]
                if not selected_columns:
                    selected_columns = filtered_df.columns[:5].tolist()
                result_df = filtered_df[selected_columns]
                return {
                    "type": "table",
                    "data": result_df.to_dict(orient="records"),
                    "columns": selected_columns,
                    "graph_data": graph_data,
                    "filtered_data": filtered_df,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": "CreatedDate field required for recent analysis"}

        elif analysis_type == "distribution":
            valid_fields = [f for f in fields if f in filtered_df.columns]
            if not valid_fields:
                return {"type": "error", "message": f"No valid fields for distribution: {fields}"}
            result_data = {}
            for field in valid_fields:
                filtered_df = filtered_df[filtered_df[field].notna() & (filtered_df[field].astype(str).str.lower() != 'none')]
                total = len(filtered_df)
                value_counts = filtered_df[field].value_counts()
                percentages = (value_counts / total * 100).round(2)
                result_data[field] = {
                    "counts": value_counts.to_dict(),
                    "percentages": percentages.to_dict()
                }
                graph_data[field] = value_counts.to_dict()
            return {
                "type": "distribution",
                "fields": valid_fields,
                "data": result_data,
                "graph_data": graph_data,
                "filtered_data": filtered_df,
                "is_product_related": is_product_related,
                "is_sales_related": is_sales_related,
                "selected_quarter": selected_quarter
            }

        elif analysis_type == "quarterly_distribution":
            if object_type in ["lead", "event", "opportunity", "task"] and 'CreatedDate' in filtered_df.columns:
                quarterly_data = {}
                quarterly_graph_data = {}
                valid_fields = [f for f in fields if f in filtered_df.columns]
                if not valid_fields:
                    quarterly_data[selected_quarter] = {}
                    logger.info(f"No valid fields for {selected_quarter}, skipping")
                    return {
                        "type": "quarterly_distribution",
                        "fields": valid_fields,
                        "data": quarterly_data,
                        "graph_data": {selected_quarter: quarterly_graph_data},
                        "filtered_data": filtered_df,
                        "is_sales_related": is_sales_related,
                        "selected_quarter": selected_quarter
                    }
                field = valid_fields[0]
                logger.info(f"Field for distribution: {field}")
                logger.info(f"Filtered DataFrame before value_counts:\n{filtered_df[field].head().to_string()}")
                dist = filtered_df[field].value_counts().to_dict()
                dist = {str(k): v for k, v in dist.items()}
                logger.info(f"Distribution for {field} in {selected_quarter}: {dist}")
                if object_type == "lead" and field == "Customer_Feedback__c":
                    if 'Interested' not in dist:
                        dist['Interested'] = 0
                    if 'Not Interested' not in dist:
                        dist['Not Interested'] = 0
                quarterly_data[selected_quarter] = dist
                quarterly_graph_data[field] = dist
                for filter_field in filters.keys():
                    if filter_field in filtered_df.columns:
                        quarterly_graph_data[filter_field] = filtered_df[filter_field].dropna().value_counts().to_dict()
                        logger.info(f"Graph data for filter field {filter_field}: {quarterly_graph_data[filter_field]}")
                graph_data = {selected_quarter: quarterly_graph_data}

                return {
                    "type": "quarterly_distribution",
                    "fields": valid_fields,
                    "data": quarterly_data,
                    "graph_data": graph_data,
                    "filtered_data": filtered_df,
                    "is_sales_related": is_sales_related,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"Quarterly distribution requires {object_type} data with CreatedDate"}

        elif analysis_type == "source_wise_lead":
            if object_type == "lead":
                required_fields = ["LeadSource"]
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}
                funnel_data = filtered_df.groupby(required_fields).size().reset_index(name="Count")
                graph_data["LeadSource"] = funnel_data.set_index("LeadSource")["Count"].to_dict()
                return {
                    "type": "source_wise_lead",
                    "fields": fields,
                    "funnel_data": funnel_data,
                    "graph_data": graph_data,
                    "filtered_data": filtered_df,
                    "is_sales_related": is_sales_related,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"Source-wise funnel not supported for {object_type}"}

        #=======================product wise funnel=============================
        elif analysis_type == "product_wise_funnel":
            if object_type == "lead":
                required_fields = ["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c", "Project_Category__c"]
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}

               
                
                filtered_events = events_df.merge(
                    filtered_df[["OwnerId", "Project_Category__c"]],
                    left_on="CreatedById",
                    right_on="OwnerId",
                    how="left"
                ).dropna(subset=["Project_Category__c"])


                for field, value in filters.items():
                    if field in filtered_events.columns:
                        if isinstance(value, str):
                            filtered_events = filtered_events[filtered_events[field] == value]
                        elif isinstance(value, dict):
                            if field == "CreatedDate":
                                if "$gte" in value:
                                    gte_value = pd.to_datetime(value["$gte"], utc=True)
                                    filtered_events = filtered_events[filtered_events[field] >= gte_value]
                                if "$lte" in value:
                                    lte_value = pd.to_datetime(value["$lte"], utc=True)
                                    filtered_events = filtered_events[filtered_events[field] <= lte_value]

                # Group by Project_Category__c
                grouped_df = filtered_df.groupby("Project_Category__c")
                product_funnel_data = {}
                product_graph_data = {}

                for product, group in grouped_df:
                    total_leads = len(group)
                    valid_leads = len(group[group["Customer_Feedback__c"] != 'Junk'])
                    sol_leads = len(group[group["Status"] == "Qualified"])
                    meeting_booked = len(group[
                        (group["Status"] == "Qualified") & (group["Is_Appointment_Booked__c"] == True)
                    ])
                    # Filter meetings for this specific product
                    # Step 1: Get OwnerIds for this product
                    owner_ids = group["OwnerId"].dropna().unique()

                    # Step 2: Filter events where CreatedById is in owner_ids
                    product_events = events_df[events_df["CreatedById"].isin(owner_ids)]

                    # Step 3: Meeting done = only those events with status 'Completed'
                    meeting_done = len(product_events[product_events["Appointment_Status__c"] == "Completed"])
                    
                    disqualified_leads = len(group[group["Customer_Feedback__c"] == "Not Interested"])
                    open_leads = len(group[group["Status"].isin(["New", "Nurturing"])])
                    junk_percentage = ((total_leads - valid_leads) / total_leads * 100) if total_leads > 0 else 0
                    tl_vl_ratio = (valid_leads / total_leads * 100) if total_leads > 0 else 0
                    vl_sol_ratio = (valid_leads / sol_leads * 100 if sol_leads > 0 else 0) if valid_leads > 0 else 0
                    sol_mb_ratio = (meeting_booked / sol_leads * 100 if sol_leads > 0 else 0) if meeting_booked > 0 else 0
                    md_sd_ratio = 0  # Initialize

                    if "CreatedDate" in filters:
                        date_filter = filters["CreatedDate"]
                        if "$gte" in date_filter:
                            gte_value = pd.to_datetime(date_filter["$gte"], utc=True)
                            opportunities_df = opportunities_df[opportunities_df["CreatedDate"] >= gte_value]
                        if "$lte" in date_filter:
                            lte_value = pd.to_datetime(date_filter["$lte"], utc=True)
                            opportunities_df = opportunities_df[opportunities_df["CreatedDate"] <= lte_value]

                    sale_done = len(opportunities_df[
                        (opportunities_df["CreatedById"].isin(owner_ids)) &
                        (opportunities_df["Sales_Order_Number__c"].notna()) & 
                        (opportunities_df["Sales_Order_Number__c"] != "None") 
                       
                    ])
                    md_sd_ratio = (meeting_done / sale_done * 100 if sale_done > 0 else 0) if meeting_done > 0 else 0

                    product_funnel_data[product] = {
                        "Total Leads": total_leads,
                        "Valid Leads": valid_leads,
                        "Sales Opportunity Leads (SOL)": sol_leads,
                        "Meeting Booked": meeting_booked,
                        "Meeting Done": meeting_done,
                        "Sale Done": sale_done,
                        "Disqualified Leads": disqualified_leads,
                        "Disqualified %": round((disqualified_leads / total_leads * 100), 2) if total_leads > 0 else 0,
                        "Open Leads": open_leads,
                        "Junk %": round(junk_percentage, 2),
                        "TL:VL Ratio (%)": round(tl_vl_ratio, 2) if tl_vl_ratio != "N/A" else 0,
                        "VL:SOL Ratio (%)": round(vl_sol_ratio, 2) if vl_sol_ratio != "N/A" else 0,
                        "SOL:MB Ratio (%)": round(sol_mb_ratio, 2) if sol_mb_ratio != "N/A" else 0,
                        "MD:SD Ratio (%)": round(md_sd_ratio, 2) if md_sd_ratio != "N/A" else 0
                    }
                    product_graph_data[product] = {
                        "Total Leads": total_leads,
                        "Valid Leads": valid_leads,
                        "Sales Opportunity Leads (SOL)": sol_leads,
                        "Meeting Booked": meeting_booked,
                        "Meeting Done": meeting_done,
                        "Sale Done": sale_done
                    }

                return {
                    "type": "product_wise_funnel",
                    "funnel_data": product_funnel_data,
                    "graph_data": {"Product Funnel Stages": product_graph_data},
                    "filtered_data": filtered_df,
                    "is_user_related": is_user_related,
                    "is_product_related": is_product_related,
                    "is_source_related": is_source_related,
                    "is_project_related": is_project_related,
                    "is_product_funnel": is_product_funnel,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"Product-wise funnel not supported for {object_type}"}
        
        
        
        #=====================================new code for the project wise funnel==================
        # Assuming this code is part of a larger function where filtered_df, events_df, opportunities_df, and other variables are defined
        elif analysis_type == "project_wise_funnel":
            if object_type == "lead":
                required_fields = ["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c", "Project__c"]
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}

                filtered_events = events_df.merge(
                    filtered_df[["OwnerId", "Project__c"]],
                    left_on="CreatedById",
                    right_on="OwnerId",
                    how="left"
                ).dropna(subset=["Project__c"])

                for field, value in filters.items():
                    if field in filtered_events.columns:
                        if isinstance(value, str):
                            filtered_events = filtered_events[filtered_events[field] == value]
                        elif isinstance(value, dict):
                            if field == "CreatedDate":
                                if "$gte" in value:
                                    gte_value = pd.to_datetime(value["$gte"], utc=True)
                                    filtered_events = filtered_events[filtered_events[field] >= gte_value]
                                if "$lte" in value:
                                    lte_value = pd.to_datetime(value["$lte"], utc=True)
                                    filtered_events = filtered_events[filtered_events[field] <= lte_value]

                # Group by Project__c
                all_projects = ["EDEN", "VERIDIA", "Wave Amore", "Wave City", "Wave Estate", 
                          "Wave Executive Floors", "WAVE GARDEN", "WAVED GARDEN", "WMMC Sec 32"]
                grouped_df = filtered_df.groupby("Project__c")
                project_funnel_data = {}
                project_graph_data = {}

                for project, group in grouped_df:
                    total_leads = len(group)
                    valid_leads = len(group[group["Customer_Feedback__c"] != 'Junk'])
                    sol_leads = len(group[group["Status"] == "Qualified"])
                    meeting_booked = len(group[
                        (group["Status"] == "Qualified") & (group["Is_Appointment_Booked__c"] == True)
                    ])
                    # Filter meetings for this specific project
                    owner_ids = group["OwnerId"].dropna().unique()
                    project_events = events_df[events_df["CreatedById"].isin(owner_ids)]
                    meeting_done = len(project_events[project_events["Appointment_Status__c"] == "Completed"])
                   
                    disqualified_leads = len(group[group["Customer_Feedback__c"] == "Not Interested"])
                    open_leads = len(group[group["Status"].isin(["New", "Nurturing"])])
                    junk_percentage = ((total_leads - valid_leads) / total_leads * 100) if total_leads > 0 else 0
                    tl_vl_ratio = (valid_leads / total_leads * 100) if total_leads > 0 else 0
                    vl_sol_ratio = (valid_leads / sol_leads * 100 if sol_leads > 0 else 0) if valid_leads > 0 else 0
                    sol_mb_ratio = (meeting_booked / sol_leads * 100 if sol_leads > 0 else 0) if meeting_booked > 0 else 0
                    md_sd_ratio = 0  # Initialize
                    
                    if "CreatedDate" in filters:
                        date_filter = filters["CreatedDate"]
                        if "$gte" in date_filter:
                            gte_value = pd.to_datetime(date_filter["$gte"], utc=True)
                            opportunities_df = opportunities_df[opportunities_df["CreatedDate"] >= gte_value]
                        if "$lte" in date_filter:
                            lte_value = pd.to_datetime(date_filter["$lte"], utc=True)
                            opportunities_df = opportunities_df[opportunities_df["CreatedDate"] <= lte_value]

                    # Step 1.3: Finally calculate
                    #sale_done = len(opportunities_df[opportunities_df["Sales_Order_Number__c"].notna() & (opportunities_df["Sales_Order_Number__c"] != "None")])

                    sale_done = len(opportunities_df[
                        (opportunities_df["CreatedById"].isin(owner_ids)) &
                        (opportunities_df["Sales_Order_Number__c"].notna()) & 
                        (opportunities_df["Sales_Order_Number__c"] != "None") 
                     
                    ])
                    md_sd_ratio = (meeting_done / sale_done * 100 if sale_done > 0 else 0) if meeting_done > 0 else 0

                    project_funnel_data[project] = {
                        "Total Leads": total_leads,
                        "Valid Leads": valid_leads,
                        "Sales Opportunity Leads (SOL)": sol_leads,
                        "Meeting Booked": meeting_booked,
                        "Meeting Done": meeting_done,
                        "Sale Done": sale_done,
                        "Disqualified Leads": disqualified_leads,
                        "Disqualified %": round((disqualified_leads / total_leads * 100), 2) if total_leads > 0 else 0,
                        "Open Leads": open_leads,
                        "Junk %": round(junk_percentage, 2),
                        "TL:VL Ratio (%)": round(tl_vl_ratio, 2) if tl_vl_ratio != "N/A" else 0,
                        "VL:SOL Ratio (%)": round(vl_sol_ratio, 2) if vl_sol_ratio != "N/A" else 0,
                        "SOL:MB Ratio (%)": round(sol_mb_ratio, 2) if sol_mb_ratio != "N/A" else 0,
                        "MD:SD Ratio (%)": round(md_sd_ratio, 2) if md_sd_ratio != "N/A" else 0
                    }
                    project_graph_data[project] = {
                        "Total Leads": total_leads,
                        "Valid Leads": valid_leads,
                        "Sales Opportunity Leads (SOL)": sol_leads,
                        "Meeting Booked": meeting_booked,
                        "Meeting Done": meeting_done,
                        "Sale Done": sale_done
                    }

                return {
                    "type": "project_wise_funnel",
                    "funnel_data": project_funnel_data,
                    "graph_data": {"Project Funnel Stages": project_graph_data},
                    "filtered_data": filtered_df,
                    "is_user_related": is_user_related,
                    "is_product_related": is_product_related,
                    "is_source_related": is_source_related,
                    "is_project_related": is_project_related,
                    "is_product_funnel": is_product_funnel,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"Project-wise funnel not supported for {object_type}"}
        
        #======================================end of project wise funnel============================
        #=====================================new code for the highest location================
       
        elif analysis_type == "location_wise_funnel":
            if object_type == "lead":
                required_fields = ["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c", "City__c", "OwnerId"]
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}

                # Group by City__c
                grouped_df = filtered_df.groupby("City__c")
                location_funnel_data = {}
                location_graph_data = {}
                filtered_df['City__c'] = filtered_df['City__c'].str.strip().str.upper()
                
                for city, group in grouped_df:
                    total_leads = len(group)
                    meeting_booked = len(group[
                        (group["Status"] == "Qualified") & (group["Is_Appointment_Booked__c"] == True)
                    ])

                    # Get OwnerIds for this city
                    city_owner_ids = group["OwnerId"].unique()

                    # Filter Opportunities for these OwnerIds
                    sale_done = len(opportunities_df[
                        (opportunities_df["Sales_Order_Number__c"].notna()) &
                        (opportunities_df["Sales_Order_Number__c"] != "None") &
                        (opportunities_df["CreatedById"].isin(city_owner_ids))
                    ])

                    # Calculate Ratio (handle division by zero safely)
                    mb_sd_ratio = round((meeting_booked / sale_done * 100), 2) if sale_done > 0 else 0

                    location_funnel_data[city] = {
                        "Meeting Booked": meeting_booked,
                        "Sale Done": sale_done,
                        "MB:SD Ratio (%)": mb_sd_ratio
                    }

                    location_graph_data[city] = {
                        "Meeting Booked": meeting_booked,
                        "Sale Done": sale_done
                    }

                return {
                    "type": "location_wise_funnel",
                    "funnel_data": location_funnel_data,
                    "graph_data": {"Location Funnel Stages": location_graph_data},
                    "filtered_data": filtered_df,
                    "is_user_related": is_user_related,
                    "is_product_related": is_product_related,
                    "is_source_related": is_source_related,
                    "is_project_related": is_project_related,
                    "is_product_funnel": is_product_funnel,
                    "selected_quarter": selected_quarter
                }

            return {"type": "error", "message": f"Location-wise funnel not supported for {object_type}"}

        #=========================================end of code================================
        
        #==================================new code for the source and user wise funnel=============
        elif analysis_type == "source_wise_funnel":
            if object_type == "lead":
                required_fields = ["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c", "LeadSource"]
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}

                
                
                filtered_events = events_df.merge(
                    filtered_df[["OwnerId", "LeadSource"]],
                    left_on="CreatedById",
                    right_on="OwnerId",
                    how="left"
                ).dropna(subset=["LeadSource"])


                for field, value in filters.items():
                    if field in filtered_events.columns:
                        if isinstance(value, str):
                            filtered_events = filtered_events[filtered_events[field] == value]
                        elif isinstance(value, dict):
                            if field == "CreatedDate":
                                if "$gte" in value:
                                    gte_value = pd.to_datetime(value["$gte"], utc=True)
                                    filtered_events = filtered_events[filtered_events[field] >= gte_value]
                                if "$lte" in value:
                                    lte_value = pd.to_datetime(value["$lte"], utc=True)
                                    filtered_events = filtered_events[filtered_events[field] <= lte_value]

                # Group by LeadSource
                grouped_df = filtered_df.groupby("LeadSource")
                source_funnel_data = {}
                source_graph_data = {}

                for source, group in grouped_df:
                    total_leads = len(group)
                    valid_leads = len(group[group["Customer_Feedback__c"] != 'Junk'])
                    sol_leads = len(group[group["Status"] == "Qualified"])
                    meeting_booked = len(group[
                        (group["Status"] == "Qualified") & (group["Is_Appointment_Booked__c"] == True)
                    ])
                    owner_ids = group["OwnerId"].dropna().unique()
                    source_events = events_df[events_df["CreatedById"].isin(owner_ids)]
                    meeting_done = len(source_events[source_events["Appointment_Status__c"] == "Completed"])
                   
                    disqualified_leads = len(group[group["Customer_Feedback__c"] == "Not Interested"])
                    open_leads = len(group[group["Status"].isin(["New", "Nurturing"])])
                    junk_percentage = ((total_leads - valid_leads) / total_leads * 100) if total_leads > 0 else 0
                    tl_vl_ratio = (valid_leads / total_leads * 100) if total_leads > 0 else 0
                    vl_sol_ratio = (valid_leads / sol_leads * 100 if sol_leads > 0 else 0) if valid_leads > 0 else 0
                    sol_mb_ratio = (meeting_booked / sol_leads * 100 if sol_leads > 0 else 0) if meeting_booked > 0 else 0
                    md_sd_ratio = 0
                    
                    #================================new code=====================
                    # ✅ Apply CreatedDate filter to opportunities_df
                    for field, value in filters.items():
                        if field == "CreatedDate" and field in opportunities_df.columns:
                            if isinstance(value, dict):
                                if "$gte" in value:
                                    gte_value = pd.to_datetime(value["$gte"], utc=True)
                                    opportunities_df = opportunities_df[opportunities_df[field] >= gte_value]
                                if "$lte" in value:
                                    lte_value = pd.to_datetime(value["$lte"], utc=True)
                                    opportunities_df = opportunities_df[opportunities_df[field] <= lte_value]
                    #==================================end of code==================

                    sale_done = len(opportunities_df[
                        (opportunities_df["CreatedById"].isin(owner_ids)) &
                        (opportunities_df["Sales_Order_Number__c"].notna()) & 
                        (opportunities_df["Sales_Order_Number__c"] != "None") 
                       
                    ])
                    md_sd_ratio = (meeting_done / sale_done * 100 if sale_done > 0 else 0) if meeting_done > 0 else 0

                    source_funnel_data[source] = {
                        "Total Leads": total_leads,
                        "Valid Leads": valid_leads,
                        "Sales Opportunity Leads (SOL)": sol_leads,
                        "Meeting Booked": meeting_booked,
                        "Meeting Done": meeting_done,
                        "Sale Done": sale_done,
                        "Disqualified Leads": disqualified_leads,
                        "Disqualified %": round((disqualified_leads / total_leads * 100), 2) if total_leads > 0 else 0,
                        "Open Leads": open_leads,
                        "Junk %": round(junk_percentage, 2),
                        "TL:VL Ratio (%)": round(tl_vl_ratio, 2) if tl_vl_ratio != "N/A" else 0,
                        "VL:SOL Ratio (%)": round(vl_sol_ratio, 2) if vl_sol_ratio != "N/A" else 0,
                        "SOL:MB Ratio (%)": round(sol_mb_ratio, 2) if sol_mb_ratio != "N/A" else 0,
                        "MD:SD Ratio (%)": round(md_sd_ratio, 2) if md_sd_ratio != "N/A" else 0
                    }
                    source_graph_data[source] = {
                        "Total Leads": total_leads,
                        "Valid Leads": valid_leads,
                        "Sales Opportunity Leads (SOL)": sol_leads,
                        "Meeting Booked": meeting_booked,
                        "Meeting Done": meeting_done,
                        "Sale Done": sale_done
                    }

                return {
                    "type": "source_wise_funnel",
                    "funnel_data": source_funnel_data,
                    "graph_data": {"Source Funnel Stages": source_graph_data},
                    "filtered_data": filtered_df,
                    "is_user_related": is_user_related,
                    "is_product_related": is_product_related,
                    "is_source_related": is_source_related,
                    "is_project_related": is_project_related,
                    "is_product_funnel": is_product_funnel,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"Source-wise funnel not supported for {object_type}"}

        #================================user wise funnel===============================
        elif analysis_type == "user_wise_funnel":
           
            if object_type == "lead":
                required_fields = ["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c", "OwnerId"]  # Adjust to your user field (e.g., CreatedById)
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}
                
                #========================new code=======================
                # ✅ CreatedDate filter leads (filtered_df) pe bhi lagao
                # ✅ Apply CreatedDate filter to events_df
                for field, value in filters.items():
                    if field == "CreatedDate" and field in events_df.columns:
                        if isinstance(value, dict):
                            if "$gte" in value:
                                gte_value = pd.to_datetime(value["$gte"], utc=True)
                                events_df = events_df[events_df[field] >= gte_value]
                            if "$lte" in value:
                                lte_value = pd.to_datetime(value["$lte"], utc=True)
                                events_df = events_df[events_df[field] <= lte_value]
                                
                # ✅ CreatedDate filter for opportunities_df
                for field, value in filters.items():
                    if field == "CreatedDate" and field in opportunities_df.columns:
                        if isinstance(value, dict):
                            if "$gte" in value:
                                gte_value = pd.to_datetime(value["$gte"], utc=True)
                                opportunities_df = opportunities_df[opportunities_df[field] >= gte_value]
                            if "$lte" in value:
                                lte_value = pd.to_datetime(value["$lte"], utc=True)
                                opportunities_df = opportunities_df[opportunities_df[field] <= lte_value]

                

                #=========================end of code=======================

                filtered_df["Id"] = filtered_df["Id"].astype(str)
                
                # ✅ Apply CreatedDate filter to opportunities_df for accurate Sale Done
                if "CreatedDate" in filters and "CreatedDate" in opportunities_df.columns:
                    created_filter = filters["CreatedDate"]
                    if isinstance(created_filter, dict):
                        if "$gte" in created_filter:
                            gte = pd.to_datetime(created_filter["$gte"], utc=True)
                            opportunities_df = opportunities_df[opportunities_df["CreatedDate"] >= gte]
                        if "$lte" in created_filter:
                            lte = pd.to_datetime(created_filter["$lte"], utc=True)
                            opportunities_df = opportunities_df[opportunities_df["CreatedDate"] <= lte]

                events_df["CreatedById"] = events_df["CreatedById"].astype(str)

                filtered_events = events_df.merge(
                    filtered_df[["Id"]],
                    left_on="CreatedById",
                    right_on="Id",
                    how="inner"
                )


                user_mapping = dict(zip(users_df['Id'], users_df['Name']))
                # Group leads by OwnerId
                #grouped_df = leads_df.groupby("OwnerId")
                grouped_df = filtered_df.groupby("OwnerId")

                user_funnel_data = {}
                user_graph_data = {}
                for user, group in grouped_df:
                    user_name = user_mapping.get(user, user)  # fallback to ID if name not found
                    user_events = events_df[events_df["CreatedById"] == user]
                    #user_events = filtered_events[filtered_events["CreatedById"] == user]

                    user_opportunities = opportunities_df[opportunities_df["CreatedById"] == user]

                    total_leads = len(group)
                    valid_leads = len(group[group["Customer_Feedback__c"] != 'Junk'])
                    sol_leads = len(group[group["Status"] == "Qualified"])
                    meeting_booked = len(group[
                        (group["Status"] == "Qualified") & (group["Is_Appointment_Booked__c"] == True)
                    ])
                    
                    
                    user_meetings = filtered_events[filtered_events["CreatedById"] == user]
                    meeting_done = len(user_events[user_events["Appointment_Status__c"] == "Completed"])
                    
                    

                    disqualified_leads = len(group[group["Customer_Feedback__c"] == "Not Interested"])
                    open_leads = len(group[group["Status"].isin(["New", "Nurturing"])])
                    junk_percentage = ((total_leads - valid_leads) / total_leads * 100) if total_leads > 0 else 0
                    tl_vl_ratio = (valid_leads / total_leads * 100) if total_leads > 0 else 0
                    vl_sol_ratio = (valid_leads / sol_leads * 100 if sol_leads > 0 else 0) if valid_leads > 0 else 0
                    sol_mb_ratio = (meeting_booked / sol_leads * 100 if sol_leads > 0 else 0) if meeting_booked > 0 else 0
                    md_sd_ratio = 0
                    sale_done = len(user_opportunities[
                        user_opportunities["Sales_Order_Number__c"].notna() &
                        (user_opportunities["Sales_Order_Number__c"] != "None") &
                        (user_opportunities["Sales_Order_Number__c"] != "")
                    ])

                    #sale_done = len(user_opportunities[user_opportunities["Sales_Order_Number__c"].notna()])
                    md_sd_ratio = (meeting_done / sale_done * 100 if sale_done > 0 else 0) if meeting_done > 0 else 0

                    # Use user_name instead of user ID
                    user_funnel_data[user_name] = {
                        "Total Leads": total_leads,
                        "Valid Leads": valid_leads,
                        "Sales Opportunity Leads (SOL)": sol_leads,
                        "Meeting Booked": meeting_booked,
                        "Meeting Done": meeting_done,
                        "Sale Done": sale_done,
                        "Disqualified Leads": disqualified_leads,
                        "Disqualified %": round((disqualified_leads / total_leads * 100), 2) if total_leads > 0 else 0,
                        "Open Leads": open_leads,
                        "Junk %": round(junk_percentage, 2),
                        "TL:VL Ratio (%)": round(tl_vl_ratio, 2),
                        "VL:SOL Ratio (%)": round(vl_sol_ratio, 2),
                        "SOL:MB Ratio (%)": round(sol_mb_ratio, 2),
                        "MD:SD Ratio (%)": round(md_sd_ratio, 2)
                    }
                    user_graph_data[user_name] = {
                        "Total Leads": total_leads,
                        "Valid Leads": valid_leads,
                        "Sales Opportunity Leads (SOL)": sol_leads,
                        "Meeting Booked": meeting_booked,
                        "Meeting Done": meeting_done,
                        "Sale Done": sale_done
                    }
                

                return {
                    "type": "user_wise_funnel",
                    "funnel_data": user_funnel_data,
                    "graph_data": {"User Funnel Stages": user_graph_data},
                    "filtered_data": filtered_df,
                    "is_user_related": is_user_related,
                    "is_product_related": is_product_related,
                    "is_source_related": is_source_related,
                    "is_project_related": is_project_related,
                    "is_product_funnel": is_product_funnel,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"User-wise funnel not supported for {object_type}"}
        
        
        #===============================end of code=============================================
        #=================================start user wise follow up===========================
        
        elif analysis_type == "user_wise_follow_up":
            if object_type == "task":
                required_fields = ["Subject", "OwnerId"]
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}

                # Ensure IDs are strings for consistent joining
                filtered_df["OwnerId"] = filtered_df["OwnerId"].astype(str)
                users_df["Id"] = users_df["Id"].astype(str)

                # Apply filters (e.g., date or other field-based filters)
                for field, value in filters.items():
                    if field in filtered_df.columns:
                        if isinstance(value, str):
                            filtered_df = filtered_df[filtered_df[field] == value]
                        elif isinstance(value, dict):
                            if field == "CreatedDate":
                               
                                if "$gte" in value:
                                    gte_value = pd.to_datetime(value["$gte"], utc=True)
                                    filtered_df = filtered_df[filtered_df[field] >= gte_value]
                                if "$lte" in value:
                                    lte_value = pd.to_datetime(value["$lte"], utc=True)
                                    filtered_df = filtered_df[filtered_df[field] <= lte_value]

                # 🔥 Identify which Subject to filter
                # By default Follow Up
                subject_filter = "Follow Up"
                
                # If "sale" keyword present, use "Sales Follow Up"
                if "sale" in analysis_type.lower():
                    subject_filter = "Sales Follow Up"

                # Apply subject filter
                filtered_df = filtered_df[filtered_df["Subject"].str.strip().str.lower() == subject_filter.lower()]

                # Map OwnerId to user names
                user_mapping = dict(zip(users_df['Id'], users_df['Name']))

                # Group tasks by OwnerId and Subject
                grouped_df = filtered_df.groupby(["OwnerId", "Subject"]).size().reset_index(name="Count")

                # Prepare result in the format: user | Subject | Count
                user_follow_up_data = {}
                for _, row in grouped_df.iterrows():
                    user_id = row["OwnerId"]
                    user_name = user_mapping.get(user_id, user_id)  # Fallback to ID if name not found
                    subject = row["Subject"]
                    count = row["Count"]
                    
                    if user_name not in user_follow_up_data:
                        user_follow_up_data[user_name] = []
                    user_follow_up_data[user_name].append({
                        "Subject": subject,
                        "Count": count
                    })

                # Prepare graph data (optional, for visualization)
                user_graph_data = {}
                for user_name, data in user_follow_up_data.items():
                    user_graph_data[user_name] = {item["Subject"]: item["Count"] for item in data}

                return {
                    "type": "user_wise_follow_up",
                    "follow_up_data": user_follow_up_data,  # Format: {user_name: [{"Subject": subject, "Count": count}, ...]}
                    "graph_data": {"User Follow-Up Stages": user_graph_data},  # For visualization
                    "filtered_data": filtered_df,
                    "is_user_related": True,
                  
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"User-wise follow-up not supported for {object_type}"}


       
        #==============================project wise follow up =========================
        elif analysis_type == "project_wise_follow_up":
            if object_type == "task":
                required_fields = ["Subject", "OwnerId"]
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}

                # Ensure IDs are strings for consistent joining
                filtered_df["OwnerId"] = filtered_df["OwnerId"].astype(str)
                leads_df["OwnerId"] = leads_df["OwnerId"].astype(str)

                # Join tasks with leads to get Project__c
                filtered_df = filtered_df.merge(
                    leads_df[["OwnerId", "Project__c"]].drop_duplicates(),
                    left_on="OwnerId",
                    right_on="OwnerId",
                    how="inner"
                )

                # Apply filters (e.g., date or other field-based filters)
                for field, value in filters.items():
                    if field in filtered_df.columns:
                        if isinstance(value, str):
                            filtered_df = filtered_df[filtered_df[field] == value]
                        elif isinstance(value, dict):
                            if field == "CreatedDate":
                                if "$gte" in value:
                                    gte_value = pd.to_datetime(value["$gte"], utc=True)
                                    filtered_df = filtered_df[filtered_df[field] >= gte_value]
                                if "$lte" in value:
                                    lte_value = pd.to_datetime(value["$lte"], utc=True)
                                    filtered_df = filtered_df[filtered_df[field] <= lte_value]

                # Filter for specific subjects: "Follow Up" or "Sales Follow Up"
                subject_filters = ["Follow Up", "Sales Follow Up"]
                # Agar user ke query me sirf 'follow up' likha hai, aur 'sales' nahi likha:
                if "follow up" in user_query.lower() and "sales" not in user_query.lower():
                    subject_filters = ["Follow Up"]
                elif "sales follow up" in user_query.lower():
                    subject_filters = ["Sales Follow Up"]
                    
                # Apply filter
                filtered_df= filtered_df[filtered_df["Subject"].str.strip().str.lower().isin([s.lower() for s in subject_filters])]
                #filtered_df = filtered_df[filtered_df["Subject"].str.strip().str.lower().isin([s.lower() for s in subject_filters])]

                # Group tasks by Project__c and Subject
                grouped_df = filtered_df.groupby(["Project__c", "Subject"]).size().reset_index(name="Count")

                # Prepare result in the format: Project | Subject | Count
                project_follow_up_data = {}
                for _, row in grouped_df.iterrows():
                    project = row["Project__c"]
                    subject = row["Subject"]
                    count = row["Count"]
                    
                    if project not in project_follow_up_data:
                        project_follow_up_data[project] = []
                    project_follow_up_data[project].append({
                        "Subject": subject,
                        "Count": count
                    })

                # Prepare graph data (optional, for visualization)
                project_graph_data = {}
                for project, data in project_follow_up_data.items():
                    project_graph_data[project] = {item["Subject"]: item["Count"] for item in data}

                return {
                    "type": "project_wise_follow_up",
                    "follow_up_data": project_follow_up_data,  # Format: {project: [{"Subject": subject, "Count": count}, ...]}
                    "graph_data": {"Project Follow-Up Stages": project_graph_data},  # For visualization
                    "filtered_data": filtered_df,
                    "is_user_related": False,
                    "is_product_related": False,
                    "is_source_related": False,
                    "is_project_related": True,
                    "is_product_funnel": False,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"Project-wise follow-up not supported for {object_type}"}

        #================================== end of user wise follow up=========================
        elif analysis_type == "product_wise_follow_up":
            if object_type == "task":
                required_fields = ["Subject", "OwnerId"]
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}

                # Ensure IDs are strings for consistent joining
                filtered_df["OwnerId"] = filtered_df["OwnerId"].astype(str)
                leads_df["OwnerId"] = leads_df["OwnerId"].astype(str)

                # Join tasks with leads to get Project_Category__c
                filtered_df = filtered_df.merge(
                    leads_df[["OwnerId", "Project_Category__c"]].drop_duplicates(),
                    left_on="OwnerId",
                    right_on="OwnerId",
                    how="inner"
                )

                # Apply filters (e.g., date or other field-based filters)
                for field, value in filters.items():
                    if field in filtered_df.columns:
                        if isinstance(value, str):
                            filtered_df = filtered_df[filtered_df[field] == value]
                        elif isinstance(value, dict):
                            if field == "CreatedDate":
                                if "$gte" in value:
                                    gte_value = pd.to_datetime(value["$gte"], utc=True)
                                    filtered_df = filtered_df[filtered_df[field] >= gte_value]
                                if "$lte" in value:
                                    lte_value = pd.to_datetime(value["$lte"], utc=True)
                                    filtered_df = filtered_df[filtered_df[field] <= lte_value]

                # Filter for specific subjects: "Follow Up" or "Sales Follow Up"
                subject_filters = ["Follow Up", "Sales Follow Up"]
                # If user query contains only 'follow up' and not 'sales'
                if "follow up" in user_query.lower() and "sales" not in user_query.lower():
                    subject_filters = ["Follow Up"]
                elif "sales follow up" in user_query.lower():
                    subject_filters = ["Sales Follow Up"]
                
                # Apply subject filter
                filtered_df = filtered_df[filtered_df["Subject"].str.strip().str.lower().isin([s.lower() for s in subject_filters])]

                # Group tasks by Project_Category__c and Subject
                grouped_df = filtered_df.groupby(["Project_Category__c", "Subject"]).size().reset_index(name="Count")

                # Prepare result in the format: Product | Subject | Count
                product_follow_up_data = {}
                for _, row in grouped_df.iterrows():
                    product = row["Project_Category__c"]
                    subject = row["Subject"]
                    count = row["Count"]
                    
                    if product not in product_follow_up_data:
                        product_follow_up_data[product] = []
                    product_follow_up_data[product].append({
                        "Subject": subject,
                        "Count": count
                    })

                # Prepare graph data (optional, for visualization)
                product_graph_data = {}
                for product, data in product_follow_up_data.items():
                    product_graph_data[product] = {item["Subject"]: item["Count"] for item in data}

                return {
                    "type": "product_wise_follow_up",
                    "follow_up_data": product_follow_up_data,  # Format: {product: [{"Subject": subject, "Count": count}, ...]}
                    "graph_data": {"Product Follow-Up Stages": product_graph_data},  # For visualization
                    "filtered_data": filtered_df,
                    "is_user_related": False,
                    "is_product_related": True,
                    "is_source_related": False,
                    "is_project_related": False,
                    "is_product_funnel": False,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"Product-wise follow-up not supported for {object_type}"}

        
        elif analysis_type == "source_wise_follow_up":
            if object_type == "task":
                required_fields = ["Subject", "OwnerId"]
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}

                # Ensure IDs are strings for consistent joining
                filtered_df["OwnerId"] = filtered_df["OwnerId"].astype(str)
                leads_df["OwnerId"] = leads_df["OwnerId"].astype(str)

                # Join tasks with leads to get LeadSource
                filtered_df = filtered_df.merge(
                    leads_df[["OwnerId", "LeadSource"]].drop_duplicates(),
                    left_on="OwnerId",
                    right_on="OwnerId",
                    how="inner"
                )

                # Apply filters (e.g., date or other field-based filters)
                for field, value in filters.items():
                    if field in filtered_df.columns:
                        if isinstance(value, str):
                            filtered_df = filtered_df[filtered_df[field] == value]
                            
                        elif isinstance(value, dict):
                            if field == "CreatedDate":
                                if "$gte" in value:
                                    gte_value = pd.to_datetime(value["$gte"], utc=True)
                                    filtered_df = filtered_df[filtered_df["CreatedDate"] >= gte_value]
                                if "$lte" in value:
                                    lte_value = pd.to_datetime(value["$lte"], utc=True)
                                    filtered_df = filtered_df[filtered_df["CreatedDate"] <= lte_value]

                # Filter for specific subjects: "Follow Up" or "Sales Follow Up"
                subject_filters = ["Follow Up", "Sales Follow Up"]
                # If user query contains only 'follow up' and not 'sales'
                if "follow up" in user_query.lower() and "sales" not in user_query.lower():
                    subject_filters = ["Follow Up"]
                elif "sales follow up" in user_query.lower():
                    subject_filters = ["Sales Follow Up"]
                
                # Apply subject filter
                filtered_df = filtered_df[filtered_df["Subject"].str.strip().str.lower().isin([s.lower() for s in subject_filters])]

                # Group tasks by LeadSource and Subject
                grouped_df = filtered_df.groupby(["LeadSource", "Subject"]).size().reset_index(name="Count")

                # Prepare result in the format: Source | Subject | Count
                source_follow_up_data = {}
                for _, row in grouped_df.iterrows():
                    source = row["LeadSource"]
                    subject = row["Subject"]
                    count = row["Count"]
                    
                    if source not in source_follow_up_data:
                        source_follow_up_data[source] = []
                    source_follow_up_data[source].append({
                        "Subject": subject,
                        "Count": count
                    })

                # Prepare graph data (optional, for visualization)
                source_graph_data = {}
                for source, data in source_follow_up_data.items():
                    source_graph_data[source] = {item["Subject"]: item["Count"] for item in data}

                return {
                    "type": "source_wise_follow_up",
                    "follow_up_data": source_follow_up_data,  # Format: {source: [{"Subject": subject, "Count": count}, ...]}
                    "graph_data": {"Source Follow-Up Stages": source_graph_data},  # For visualization
                    "filtered_data": filtered_df,
                    "is_user_related": False,
                    "is_product_related": False,
                    "is_source_related": True,
                    "is_project_related": False,
                    "is_product_funnel": False,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"Source-wise follow-up not supported for {object_type}"}
            
        #============================= new code for the not open lead========================
       

        elif analysis_type == "open_lead_not_follow_up":
            if object_type == "task":
                required_fields = ["Subject", "OwnerId"]
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}

                # Ensure IDs are strings for consistent joining
                filtered_df["OwnerId"] = filtered_df["OwnerId"].astype(str)
                leads_df["OwnerId"] = leads_df["OwnerId"].astype(str)

                print(f"✅ Starting filter loop, filters received: {filtered_df}")
                print(f"✅ Columns available in filtered_df: {list(filtered_df.columns)}")
                for field, value in filters.items():
                    if field in filtered_df.columns:
                        if isinstance(value, str):
                            filtered_df = filtered_df[filtered_df[field] == value]
                            print(f"Applied string filter: {field} = {value}, records: {len(filtered_df)}")
                        elif isinstance(value, dict):
                            # Enhanced date filtering for CreatedDate
                            if field == "CreatedDate":
                                print(f"Processing date filter for {field}: {value}")
                                if "$gte" in value:
                                    try:
                                        gte_value = pd.to_datetime(value["$gte"], utc=True)
                                        filtered_df = filtered_df[filtered_df[field] >= gte_value]
                                        print(f"✅ Applied start date filter: {field} >= {gte_value}")
                                        print(f"   Records after start date filter: {len(filtered_df)}")
                                    except Exception as e:
                                        print(f"❌ Error parsing start date: {value['$gte']}, error: {e}")
                                        return {"type": "error", "message": f"Invalid start date format: {value['$gte']}"}
                                if "$lte" in value:
                                    try:
                                        lte_value = pd.to_datetime(value["$lte"], utc=True)
                                        filtered_df = filtered_df[filtered_df[field] <= lte_value]
                                        print(f"✅ Applied end date filter: {field} <= {lte_value}")
                                        print(f"   Records after end date filter: {len(filtered_df)}")
                                    except Exception as e:
                                        print(f"❌ Error parsing end date: {value['$lte']}, error: {e}")
                                        return {"type": "error", "message": f"Invalid end date format: {value['$lte']}"}
                                
                                # Log the date range being applied
                                if "$gte" in value and "$lte" in value:
                                    print(f"📅 Date range filter applied: {field} between {value['$gte']} and {value['$lte']}")
                                    print(f"📊 Total records after date range filter: {len(filtered_df)}")
                                    
                                    # Show sample dates to verify filtering
                                    if len(filtered_df) > 0:
                                        sample_dates = filtered_df[field].head(3).tolist()
                                        print(f"📋 Sample dates in filtered data: {sample_dates}")
                            else:
                                print(f"Applied dict filter for {field}: {value}")
                        else:
                            print(f"Applied other filter for {field}: {value}")
                    else:
                        print(f"⚠️ Field {field} not found in columns: {list(filtered_df.columns)}")

                # Merge to get Customer_Feedback__c from leads
                print(f"Task count: {len(filtered_df)}, Lead count: {len(leads_df)}")
                print(f"Leads_df columns: {list(leads_df.columns)}")
                
                # Check if Customer_Feedback__c exists in leads_df
                if "Customer_Feedback__c" not in leads_df.columns:
                    return {"type": "error", "message": "leads_df is missing the 'Customer_Feedback__c' column"}
                
                # Remove duplicates before merge
                #filtered_df = filtered_df.drop_duplicates(subset=["OwnerId"])
                print(f"task filter data ++++", filtered_df)
                leads_df_unique = leads_df.groupby("OwnerId", as_index=False).agg({
                    "Customer_Feedback__c": "first"  # Ya most relevant logic
                })
                # Perform the merge
                try:
                    merged_df = filtered_df.merge(
                        leads_df_unique[["OwnerId", "Customer_Feedback__c"]],
                        on="OwnerId",
                        how="inner"
                    )
                    print(f"Merged DataFrame shape: {merged_df.shape}")
                    print(f"Merged DataFrame columns: {list(merged_df.columns)}")
                except Exception as e:
                    print(f"Merge error: {str(e)}")
                    return {"type": "error", "message": f"Error during merge: {str(e)}"}

                # Handle duplicate column names from merge
               
                if "Customer_Feedback__c_x" in merged_df.columns and "Customer_Feedback__c_y" in merged_df.columns:
                    # Use the one from leads_df (usually _y suffix)
                    merged_df["Customer_Feedback__c"] = merged_df["Customer_Feedback__c_y"]
                    merged_df = merged_df.drop(["Customer_Feedback__c_x", "Customer_Feedback__c_y"], axis=1)
                    print("Fixed duplicate Customer_Feedback__c columns")
                elif "Customer_Feedback__c_x" in merged_df.columns:
                    merged_df["Customer_Feedback__c"] = merged_df["Customer_Feedback__c_x"]
                    merged_df = merged_df.drop("Customer_Feedback__c_x", axis=1)
                    print("Fixed Customer_Feedback__c_x column")
                elif "Customer_Feedback__c_y" in merged_df.columns:
                    merged_df["Customer_Feedback__c"] = merged_df["Customer_Feedback__c_y"]
                    merged_df = merged_df.drop("Customer_Feedback__c_y", axis=1)
                    print("Fixed Customer_Feedback__c_y column")
                
                print(f"Final columns after fix: {list(merged_df.columns)}")

                # Filter for tasks where Subject is not 'Follow Up' or 'Sales Follow Up'
                excluded_subjects = ["Follow Up"]
                merged_df = merged_df[~merged_df["Subject"].str.strip().str.lower().isin([s.lower() for s in excluded_subjects])]
                
                print(f"After subject filter: {len(merged_df)} rows")

                # Filter out None/null Customer_Feedback__c values
                merged_df = merged_df[
                    (merged_df["Customer_Feedback__c"].astype(str).str.lower().str.strip() == "discussion pending") |
                    (merged_df["Customer_Feedback__c"].astype(str).str.lower().str.strip() == "none") |
                    (merged_df["Customer_Feedback__c"].isna())
                   
                ]
                

                # Group by Customer_Feedback__c and Subject
                grouped_df = merged_df.groupby(["Customer_Feedback__c", "Subject"]).size().reset_index(name="Count")
                
                print(f"Grouped data shape: {grouped_df.shape}")

                # Prepare the result: Customer | Subject | Count
                open_lead_data = {}
                for _, row in grouped_df.iterrows():
                    customer = row["Customer_Feedback__c"] if pd.notna(row["Customer_Feedback__c"]) else "None"
                    subject = row["Subject"]
                    count = row["Count"]

                    if customer not in open_lead_data:
                        open_lead_data[customer] = []
                    open_lead_data[customer].append({
                        "Subject": subject,
                        "Count": count
                    })

                # Prepare graph data (optional)
                open_lead_graph = {}
                for customer, data in open_lead_data.items():
                    open_lead_graph[customer] = {item["Subject"]: item["Count"] for item in data}

                return {
                    "type": "open_lead_not_follow_up",
                    "open_lead_data": open_lead_data,  # Format: {customer: [{"Subject": subject, "Count": count}, ...]}
                    "graph_data": {"Open Lead Without Follow-Up": open_lead_graph},  # For visualization
                    "filtered_data": merged_df,
                    "selected_quarter": selected_quarter
                }

            return {"type": "error", "message": f"Open lead without follow-up not supported for {object_type}"}


        #==============================end of the product and source wise follow up===========
        
        #==================================start crm _team_member================
      
        elif analysis_type == "crm_team_member":
           
            if object_type == "lead":
                required_fields = ["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c", "OwnerId"]  # Adjust to your user field (e.g., CreatedById)
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                
                if missing_fields:
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}

                filtered_df["Id"] = filtered_df["Id"].astype(str)
                events_df["CreatedById"] = events_df["CreatedById"].astype(str)

                filtered_events = events_df.merge(
                    filtered_df[["Id"]].drop_duplicates(),
                    left_on="CreatedById",
                    right_on="Id",
                    how="inner"
                )

                                
                # Apply filters
                for field, value in filters.items():
                    if field in filtered_df.columns:

                        # ✅ Handle Regex Dict filter
                        if isinstance(value, dict) and "$regex" in value:
                            pattern = value["$regex"]
                            case_insensitive = value.get("options", "") == "i"
                            
                            if case_insensitive:
                                filtered_df = filtered_df[filtered_df[field].str.contains(pattern, case=False, na=False, regex=True)]
                            else:
                                filtered_df = filtered_df[filtered_df[field].str.contains(pattern, case=True, na=False, regex=True)]

                        # Normal string filter
                        elif isinstance(value, str):
                            filtered_df = filtered_df[filtered_df[field] == value]

                        # Date filter handling
                        elif isinstance(value, dict) and field == "CreatedDate":
                            if "$gte" in value:
                                gte_value = pd.to_datetime(value["$gte"], utc=True)
                                filtered_df = filtered_df[filtered_df[field] >= gte_value]
                            if "$lte" in value:
                                lte_value = pd.to_datetime(value["$lte"], utc=True)
                                filtered_df = filtered_df[filtered_df[field] <= lte_value]

                user_mapping = dict(zip(users_df['Id'], users_df['Name']))
                # Group leads by OwnerId
                grouped_df = leads_df.groupby("OwnerId")
                user_funnel_data = {}
                user_graph_data = {}
                for user, group in grouped_df:
                    user_name = user_mapping.get(user, user)  # fallback to ID if name not found
                   
                    total_leads = len(group)
                   
                    sol_leads = len(group[group["Status"] == "Qualified"])
                    tl_sol_ratio = (total_leads / sol_leads * 100 if sol_leads > 0 else 0) if total_leads > 0 else 0
                    
                    # Use user_name instead of user ID
                    user_funnel_data[user_name] = {
                        "Total Leads": total_leads,
                    
                        "Sales Opportunity Leads (SOL)": sol_leads,
                       
                       
                        "TL:SOL Ratio (%)": round(tl_sol_ratio, 2)
                    }
                    user_graph_data[user_name] = {
                        "Total Leads": total_leads,
                      
                        "Sales Opportunity Leads (SOL)": sol_leads,
                        
                    }
                

                return {
                    "type": "crm_team_member",
                    "funnel_data": user_funnel_data,
                    "graph_data": {"CRM Funnel Stages": user_graph_data},
                    "filtered_data": filtered_df,
                    "is_user_related": is_user_related,
                    "is_product_related": is_product_related,
                    "is_source_related": is_source_related,
                    "is_project_related": is_project_related,
                    "is_product_funnel": is_product_funnel,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"CRM-wise funnel not supported for {object_type}"}
        
        #====================================end of code of crm team member==================
        
        elif analysis_type == "product_wise_lead":
            if object_type == "lead":
                required_fields = ["Project_Category__c"]
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}
                funnel_data = filtered_df.groupby("Project_Category__c").size().reset_index(name="Count")
                graph_data["Project_Category__c"] = funnel_data.set_index("Project_Category__c")["Count"].to_dict()
                return {
                    "type": "product_wise_lead",
                    "fields": fields,
                    "funnel_data": funnel_data,
                    "graph_data": graph_data,
                    "filtered_data": filtered_df,
                    "is_product_related": is_product_related,
                    "is_source_related": is_source_related,
                    "is_project_related": is_project_related,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"Product-wise lead not supported for {object_type}"}
        #======================end of code========================================================
        # Handle conversion funnel analysis
        elif analysis_type == "conversion_funnel":
            if object_type == "lead":
                required_fields = ["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c"]
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}

                filtered_events = events_df.copy()
                for field, value in filters.items():
                    if field in filtered_events.columns:
                        if isinstance(value, str):
                            filtered_events = filtered_events[filtered_events[field] == value]
                        elif isinstance(value, dict):
                            if field == "CreatedDate":
                                if "$gte" in value:
                                    gte_value = pd.to_datetime(value["$gte"], utc=True)
                                    filtered_events = filtered_events[filtered_events[field] >= gte_value]
                                if "$lte" in value:
                                    lte_value = pd.to_datetime(value["$lte"], utc=True)
                                    filtered_events = filtered_events[filtered_events[field] <= lte_value]

                total_leads = len(filtered_df)
                valid_leads = len(filtered_df[filtered_df["Customer_Feedback__c"] != 'Junk'])
                sol_leads = len(filtered_df[filtered_df["Status"] == "Qualified"])
                meeting_booked = len(filtered_df[
                    (filtered_df["Status"] == "Qualified") & (filtered_df["Is_Appointment_Booked__c"] == True)
                ])
                # Step 2.1: Apply date filter to events
                if "CreatedDate" in filters:
                    date_filter = filters["CreatedDate"]
                    if "$gte" in date_filter:
                        gte_value = pd.to_datetime(date_filter["$gte"], utc=True)
                        filtered_events = filtered_events[filtered_events["CreatedDate"] >= gte_value]
                    if "$lte" in date_filter:
                        lte_value = pd.to_datetime(date_filter["$lte"], utc=True)
                        filtered_events = filtered_events[filtered_events["CreatedDate"] <= lte_value]

                # Step 2.2: Then count
                meeting_done = len(filtered_events[filtered_events["Appointment_Status__c"] == "Completed"])

                #meeting_done = len(filtered_events[filtered_events["Appointment_Status__c"] == "Completed"])
                disqualified_leads = len(filtered_df[filtered_df["Customer_Feedback__c"] == "Not Interested"])
                open_leads = len(filtered_df[filtered_df["Status"].isin(["New", "Nurturing"])])
                junk_percentage = ((total_leads - valid_leads) / total_leads * 100) if total_leads > 0 else 0
                
                vl_sol_ratio = (valid_leads / sol_leads) if sol_leads > 0 else "N/A"
                tl_vl_ratio  = (valid_leads / total_leads) if total_leads > 0 else "N/A"
                sol_mb_ratio = (sol_leads / meeting_booked) if meeting_booked > 0 else "N/A"
                meeting_booked_meeting_done = (meeting_done / meeting_booked) if meeting_done > 0 else "N/A"
                # sale_done = len(opportunities_df[opportunities_df["Sales_Order_Number__c"].notna() & (opportunities_df["Sales_Order_Number__c"] != "None")])
                # Step 1.2: Apply date filters if present
                if "CreatedDate" in filters:
                    date_filter = filters["CreatedDate"]
                    if "$gte" in date_filter:
                        gte_value = pd.to_datetime(date_filter["$gte"], utc=True)
                        opportunities_df = opportunities_df[opportunities_df["CreatedDate"] >= gte_value]
                    if "$lte" in date_filter:
                        lte_value = pd.to_datetime(date_filter["$lte"], utc=True)
                        opportunities_df = opportunities_df[opportunities_df["CreatedDate"] <= lte_value]

                # Step 1.3: Finally calculate
                sale_done = len(opportunities_df[opportunities_df["Sales_Order_Number__c"].notna() & (opportunities_df["Sales_Order_Number__c"] != "None")])
                #sale_done = len(filtered_opps)
                md_sd_ratio = (meeting_done / sale_done) if sale_done > 0 else "N/A"

                # Apply user-based filtering if specified
                funnel_metrics = {
                    "TL:VL Ratio": round(tl_vl_ratio, 2) if isinstance(tl_vl_ratio, (int, float)) else tl_vl_ratio,
                    "VL:SOL Ratio": round(vl_sol_ratio, 2) if isinstance(vl_sol_ratio, (int, float)) else vl_sol_ratio,
                    "SOL:MB Ratio": round(sol_mb_ratio, 2) if isinstance(sol_mb_ratio, (int, float)) else sol_mb_ratio,
                    "MB:MD Ratio": round(meeting_booked_meeting_done, 2) if isinstance(meeting_booked_meeting_done, (int, float)) else meeting_booked_meeting_done,
                    "MD:SD Ratio": round(md_sd_ratio, 2) if isinstance(md_sd_ratio, (int, float)) else md_sd_ratio,
                    "Total Leads": total_leads,
                    "Valid Leads": valid_leads,
                    "Sales Opportunity Leads (SOL)": sol_leads,
                    "Meeting Booked": meeting_booked,
                    "Meeting Done": meeting_done,
                    "Sale Done": sale_done,
                }
                graph_data["Funnel Stages"] = {
                    "Total Leads": total_leads,
                    "Valid Leads": valid_leads,
                    "Sales Opportunity Leads (SOL)": sol_leads,
                    "Meeting Booked": meeting_booked,
                    "Meeting Done": meeting_done,
                    "Sale Done": sale_done,
                }
                return {
                    "type": "conversion_funnel",
                    "funnel_metrics": funnel_metrics,
                    "quarterly_data": {selected_quarter: {
                        "Total Leads": total_leads,
                        "Valid Leads": valid_leads,
                        "Sales Opportunity Leads (SOL)": sol_leads,
                        "Meeting Booked": meeting_booked,
                        "Disqualified Leads": disqualified_leads,
                        "Disqualified %": round((disqualified_leads / total_leads * 100), 2) if total_leads > 0 else 0,
                        "Open Leads": open_leads,
                        "Junk %": round(junk_percentage, 2),
                        "VL:SOL Ratio": round(vl_sol_ratio, 2) if isinstance(vl_sol_ratio, (int, float)) else vl_sol_ratio,
                        "SOL:MB Ratio": round(sol_mb_ratio, 2) if isinstance(sol_mb_ratio, (int, float)) else sol_mb_ratio,
                        "MD:SD Ratio": round(md_sd_ratio, 2) if isinstance(md_sd_ratio, (int, float)) else md_sd_ratio
                    }},
                    "graph_data": graph_data,
                    "filtered_data": filtered_df,
                    "is_user_related": is_user_related,
                    "is_product_related": is_product_related,
                    "is_source_related": is_source_related,
                    "is_project_related": is_project_related,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"Conversion funnel not supported for {object_type}"}

     
      
      
      #=========================== end of code===========================================
      
       
        elif analysis_type == "Total_Appointment":
            if object_type == "event":
                required_fields = ["Appointment_Status__c"]
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    logger.error(f"Missing fields for conversion_funnel: {missing_fields}")
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}

                # Calculate Appointment Status Counts
                if 'Appointment_Status__c' in filtered_events.columns:
                    appointment_status_counts = filtered_events['Appointment_Status__c'].value_counts().to_dict()
                    logger.info(f"Appointment Status counts: {appointment_status_counts}")
                else:
                    appointment_status_counts = {}
                    logger.warning("Status column not found in filtered_events")

            return {"type": "error", "message": f" Total Appointments for {object_type}"}

        elif analysis_type == "percentage":
            if object_type in ["lead", "event", "opportunity", "task"]:
                total_records = len(df)
                percentage = (len(filtered_df) / total_records * 100) if total_records > 0 else 0
                # Custom label for disqualification percentage
                if "Customer_Feedback__c" in filters and filters["Customer_Feedback__c"] == "Not Interested":
                    label = "Percentage of Disqualified Leads"
                else:
                    label = "Percentage of " + " and ".join([f"{FIELD_DISPLAY_NAMES.get(f, f)} = {v}" for f, v in filters.items()])
                graph_data["Percentage"] = {"Matching Records": percentage, "Non-Matching Records": 100 - percentage}
                return {
                    "type": "percentage",
                    "value": round(percentage, 1),
                    "label": label,
                    "graph_data": graph_data,
                    "filtered_data": filtered_df,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"Percentage analysis not supported for {object_type}"}

        elif analysis_type == "top":
            valid_fields = [f for f in fields if f in df.columns]
            if not valid_fields:
                return {"type": "error", "message": f"No valid fields for top values: {fields}"}
            result_data = {field: filtered_df[field].value_counts().head(5).to_dict() for field in valid_fields}
            for field in valid_fields:
                graph_data[field] = filtered_df[field].value_counts().head(5).to_dict()
            return {
                "type": "distribution",
                "fields": valid_fields,
                "data": result_data,
                "graph_data": graph_data,
                "filtered_data": filtered_df,
                "is_sales_related": is_sales_related,
                "selected_quarter": selected_quarter
            }

        return {"type": "info", "message": analysis_plan.get("explanation", "Analysis completed")}

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return {"type": "error", "message": f"Analysis failed: {str(e)}"}

def render_graph(graph_data, relevant_fields, title_suffix="", quarterly_data=None):
    logger.info(f"Rendering graph with data: {graph_data}, relevant fields: {relevant_fields}")
    if not graph_data:
        st.info("No data available for graph.")
        return
    for field in relevant_fields:
        if field not in graph_data:
            logger.warning(f"No graph data for field: {field}")
            continue
        data = graph_data[field]
        if not data:
            logger.warning(f"Empty graph data for field: {field}")
            continue

        # Special handling for opportunity_vs_lead
        if field == "Opportunity vs Lead":
            try:
                plot_data = [{"Category": k, "Count": v} for k, v in data.items() if k is not None and not pd.isna(k)]
                if not plot_data:
                    st.info("No valid data for Opportunity vs Lead graph.")
                    continue
                plot_df = pd.DataFrame(plot_data)
                plot_df = plot_df.sort_values(by="Count", ascending=False)
                fig = px.bar(
                    plot_df,
                    x="Count",
                    y="Category",
                    orientation='h',
                    title=f"Opportunity vs Lead Distribution{title_suffix}",
                    color="Category",
                    color_discrete_map={
                        "Total Leads": "#1f77b4",
                        "Opportunities": "#ff7f0e"
                    }
                )
                fig.update_layout(xaxis_title="Count", yaxis_title="Category")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                logger.error(f"Error rendering Opportunity vs Lead graph: {e}")
                st.error(f"Failed to render Opportunity vs Lead graph: {str(e)}")
                
        elif field == "Funnel Stages":  # Special handling for conversion funnel
            # Filter funnel stages to match the fields in quarterly_data (used in the table)
            if quarterly_data is None:
                logger.warning("quarterly_data not provided for conversion funnel")
                st.info("Cannot render funnel graph: missing quarterly data.")
                continue
            # Get the stages from quarterly_data that match the table
            table_stages = list(quarterly_data.keys())
            # Only include stages that are both in graph_data and quarterly_data
            filtered_funnel_data = {stage: data[stage] for stage in data if stage in ["Total Leads", "Valid Leads", "Sales Opportunity Leads (SOL)", "Meeting Booked", "Meeting Done", "Sale Done"]}
            if not filtered_funnel_data:
                logger.warning("No matching funnel stages found between graph_data and table data")
                st.info("No matching data for funnel graph.")
                continue
            plot_df = pd.DataFrame.from_dict(filtered_funnel_data, orient='index', columns=['Count']).reset_index()
            plot_df.columns = ["Stage", "Count"]
            try:
                fig = go.Figure(go.Funnel(
                    y=plot_df["Stage"],
                    x=plot_df["Count"],
                    textinfo="value+percent initial",
                    marker={"color": "#1f77b4"}
                ))
                fig.update_layout(title=f"Lead Conversion Funnel{title_suffix}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                logger.error(f"Error rendering Plotly funnel chart: {e}")
                st.error(f"Failed to render graph: {str(e)}")
                
        elif field == "Product Funnel Stages":
            for product, stages in data.items():
                plot_df = pd.DataFrame.from_dict(stages, orient='index', columns=['Count']).reset_index()
                plot_df.columns = ["Stage", "Count"]
                plot_df = plot_df[plot_df["Stage"].isin(["Total Leads", "Valid Leads", "Sales Opportunity Leads (SOL)", "Meeting Booked", "Meeting Done", "Sale Done"])]
                try:
                    fig = go.Figure(go.Funnel(
                        y=plot_df["Stage"],
                        x=plot_df["Count"],
                        textinfo="value+percent initial",
                        marker={"color": "#1f77b4"}
                    ))
                    fig.update_layout(title=f"Product-Wise Funnel for {product}{title_suffix}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error rendering Plotly product funnel chart for {product}: {e}")
                    st.error(f"Failed to render product funnel graph for {product}: {str(e)}")
        
        else:
            plot_data = [{"Category": str(k), "Count": v} for k, v in data.items() if k is not None and not pd.isna(k)]
            if not plot_data:
                st.info(f"No valid data for graph for {FIELD_DISPLAY_NAMES.get(field, field)}.")
                continue
            plot_df = pd.DataFrame(plot_data)
            plot_df = plot_df.sort_values(by="Count", ascending=False)
            try:
                fig = px.bar(
                    plot_df,
                    x="Category",
                    y="Count",
                    title=f"Distribution of {FIELD_DISPLAY_NAMES.get(field, field)}{title_suffix}",
                    color="Category"
                )
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                logger.error(f"Error rendering Plotly chart: {e}")
                st.error(f"Failed to render graph: {str(e)}")

def display_analysis_result(result, analysis_plan=None, user_question=""):
    """
    Display the analysis result using Streamlit, including tables, metrics, and graphs.
    """
    result_type = result.get("type", "")
    object_type = analysis_plan.get("object_type", "lead") if analysis_plan else "lead"
    is_product_related = result.get("is_product_related", False)
    is_sales_related = result.get("is_sales_related", False)
    is_product_funnel = result.get("is_product_funnel", False)
    selected_quarter = result.get("selected_quarter", None)
    graph_data = result.get("graph_data", {})
    filtered_data = result.get("filtered_data", pd.DataFrame())

    logger.info(f"Displaying result for type: {result_type}, user question: {user_question}")

    if analysis_plan and analysis_plan.get("filters"):
        st.info(f"Filters applied: {analysis_plan['filters']}")

    def prepare_filtered_display_data(filtered_data, analysis_plan):
        if filtered_data.empty:
            logger.warning("Filtered data is empty for display")
            return pd.DataFrame(), []
        display_cols = []
        prioritized_cols = []
        if analysis_plan and analysis_plan.get("filters"):
            for field in analysis_plan["filters"]:
                if field in filtered_data.columns and field not in prioritized_cols:
                    prioritized_cols.append(field)
        if analysis_plan and analysis_plan.get("fields"):
            for field in analysis_plan["fields"]:
                if field in filtered_data.columns and field not in prioritized_cols:
                    prioritized_cols.append(field)
        display_cols.extend(prioritized_cols)
        preferred_cols = (
            ['Id', 'Name', 'Phone__c', 'LeadSource', 'Status', 'CreatedDate', 'Customer_Feedback__c']
            if object_type == "lead"
            else ['Service_Request_Number__c', 'Type', 'Subject', 'CreatedDate']
            if object_type == "case"
            else ['Id', 'Subject', 'StartDateTime', 'EndDateTime', 'Appointment_Status__c', 'CreatedDate']
            if object_type == "event"
            else ['Id', 'Name', 'StageName', 'Amount', 'CloseDate', 'CreatedDate', 'Project_Category__c', 'Sales_Order_Number__c']
            if object_type == "opportunity"
            else ['Id', 'Subject', 'Transfer_Status__c', 'Customer_Feedback__c', 'Sales_Team_Feedback__c', 'Status', 'Follow_Up_Status__c']
            if object_type == "task"
            else ['Id', 'Lead_Id__c', 'Opp_Lead_Id__c', 'Transfer_Status__c', 'Customer_Feedback__c', 'CreatedDate',
                       'Sales_Team_Feedback__c', 'Status', 'Follow_Up_Status__c', 'Subject', 'OwnerId']
        )
        max_columns = 10
        remaining_slots = max_columns - len(prioritized_cols)
        for col in preferred_cols:
            if col in filtered_data.columns and col not in display_cols and remaining_slots > 0:
                display_cols.append(col)
                remaining_slots -= 1
        display_data = filtered_data[display_cols].rename(columns=FIELD_DISPLAY_NAMES)
        return display_data, display_cols

    title_suffix = ""
    if result_type in ["quarterly_distribution", "product_wise_lead"] and selected_quarter:
        normalized_quarter = selected_quarter.strip()
        title_suffix = f" in {normalized_quarter}"
        logger.info(f"Selected quarter for display: '{normalized_quarter}' (length: {len(normalized_quarter)})")
        logger.info(f"Selected quarter bytes: {list(normalized_quarter.encode('utf-8'))}")
    else:
        logger.info(f"No quarter selected or not applicable for result_type: {result_type}")
        normalized_quarter = selected_quarter

    logger.info(f"Graph data: {graph_data}")

    # Handle opportunity_vs_lead result type
    if result_type == "opportunity_vs_lead":
        logger.info("Rendering opportunity vs lead summary")
        st.subheader(f"Opportunity vs Lead Summary{title_suffix}")
        df = pd.DataFrame(result["data"])
        st.dataframe(df.rename(columns=FIELD_DISPLAY_NAMES), use_container_width=True, hide_index=True)

    elif result_type == "metric":
        logger.info("Rendering metric result")
        st.metric(result.get("label", "Result"), f"{result.get('value', 0)}")

    elif result_type == "disqualification_summary":
        logger.info("Rendering disqualification summary")
        st.subheader(f"Disqualification Reasons Summary{title_suffix}")
        df = pd.DataFrame(result["data"])
        st.dataframe(df, use_container_width=True, hide_index=True)

    elif result_type == "junk_reason_summary":
        logger.info("Rendering junk reason summary")
        st.subheader(f"Junk Reason Summary{title_suffix}")
        df = pd.DataFrame(result["data"])
        st.dataframe(df, use_container_width=True)

    elif result_type == "conversion_funnel":
        logger.info("Rendering conversion funnel")
        funnel_metrics = result.get("funnel_metrics", {})
        quarterly_data = result.get("quarterly_data", {}).get(selected_quarter, {})
        appointment_status_counts = result.get("appointment_status_counts", 0)
        st.subheader(f"Lead Conversion Funnel Analysis{title_suffix}")
        st.info(f"Found {len(filtered_data)} leads matching the criteria.")

        # Display Appointment Status Counts as a table
        if appointment_status_counts:
            st.subheader("Appointment Status Counts")
            status_df = pd.DataFrame.from_dict(appointment_status_counts, orient='index', columns=['Count']).reset_index()
            status_df.columns = ["Appointment Status", "Count"]
            status_df = status_df.sort_values(by="Count", ascending=False)
            st.dataframe(status_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No appointment status data available.")

        # Display the funnel metrics table (ratios)
        if funnel_metrics:
            st.subheader("Funnel Metrics")
            metrics_df = pd.DataFrame.from_dict(funnel_metrics, orient='index', columns=['Value']).reset_index()
            metrics_df.columns = ["Metric", "Value"]
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    
    #=====================================prodcut wise funnel=========================
    elif result_type == "product_wise_funnel":
        logger.info("Rendering product-wise funnel")
        funnel_data = result.get("funnel_data", {})
        st.subheader(f"Product-Wise Funnel Analysis{title_suffix}")
        st.info(f"Found {len(filtered_data)} leads matching the criteria.")

        if not funnel_data:
            st.warning("No funnel data available for any product.")
            return

        # Prepare data for a single table
        metrics_list = []
        products = list(funnel_data.keys())
        if products:
            # Get all unique metrics from the first product's data (assuming all products have the same metrics)
            all_metrics = list(funnel_data[products[0]].keys())
            for metric in all_metrics:
                row = {"Metric": metric}
                for product in products:
                    row[product] = funnel_data[product][metric]
                metrics_list.append(row)

            # Create a DataFrame
            funnel_df = pd.DataFrame(metrics_list)
            # Ensure numeric columns are formatted properly
            for product in products:
                funnel_df[product] = pd.to_numeric(funnel_df[product], errors='ignore')
            funnel_df = funnel_df.round(2)  # Round to 2 decimal places for percentages and ratios

            # Display the table
            st.dataframe(funnel_df, use_container_width=True, hide_index=True)

        if st.button("Show Data", key=f"product_funnel_data_{result_type}_{selected_quarter}"):
            st.write(f"Filtered {object_type.capitalize()} Data")
            display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
            st.dataframe(display_data, use_container_width=True, hide_index=True)

    #===================================project wise funnel============================
    elif result_type == "project_wise_funnel":
        logger.info("Rendering project-wise funnel")
        funnel_data = result.get("funnel_data", {})
        st.subheader(f"Project-Wise Funnel Analysis{title_suffix}")
        st.info(f"Found {len(filtered_data)} leads matching the criteria.")

        if not funnel_data:
            st.warning("No funnel data available for any project.")
            return

        # Prepare data for a single table
        metrics_list = []
        projects = list(funnel_data.keys())
        if projects:
            # Get all unique metrics from the first project's data (assuming all projects have the same metrics)
            all_metrics = list(funnel_data[projects[0]].keys())
            for metric in all_metrics:
                row = {"Metric": metric}
                for project in projects:
                    row[project] = funnel_data[project][metric]
                metrics_list.append(row)

            # Create a DataFrame
            funnel_df = pd.DataFrame(metrics_list)
            # Ensure numeric columns are formatted properly
            for project in projects:
                funnel_df[project] = pd.to_numeric(funnel_df[project], errors='ignore')
            funnel_df = funnel_df.round(2)  # Round to 2 decimal places for percentages and ratios

            # Display the table
            st.dataframe(funnel_df, use_container_width=True, hide_index=True)

        if st.button("Show Data", key=f"project_funnel_data_{result_type}_{selected_quarter}"):
            st.write(f"Filtered {object_type.capitalize()} Data")
            display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
            st.dataframe(display_data, use_container_width=True, hide_index=True)
    
    
    #=====================================end of project wise funnel=======================
    #==================================highest location of the conversion===============
    elif result_type == "location_wise_funnel":
        logger.info("Rendering location-wise funnel")
        funnel_data = result.get("funnel_data", {})
        st.subheader(f"Location-Wise Funnel Analysis{title_suffix}")
        st.info(f"Found {len(filtered_data)} leads matching the criteria.")

        if not funnel_data:
            st.warning("No funnel data available for any location.")
            return

        # Prepare data for a single table
        metrics_list = []
        locations = list(funnel_data.keys())
        if locations:
            
            required_metrics = ["Meeting Booked", "Sale Done", "MB:SD Ratio (%)"]
            
            for metric in required_metrics:
                row = {"Metric": metric}
                for location in locations:
                    row[location] = funnel_data[location][metric]
                metrics_list.append(row)

            # Create a DataFrame
            funnel_df = pd.DataFrame(metrics_list)
            # Ensure numeric columns are formatted properly
            for location in locations:
                funnel_df[location] = pd.to_numeric(funnel_df[location], errors='ignore')
            funnel_df = funnel_df.round(2)  # Round to 2 decimal places for percentages and ratios

            # Display the table
            st.dataframe(funnel_df, use_container_width=True, hide_index=True)

        if st.button("Show Data", key=f"location_funnel_data_{result_type}_{selected_quarter}"):
            st.write(f"Filtered {object_type.capitalize()} Data")
            display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
            st.dataframe(display_data, use_container_width=True, hide_index=True)
    
    #=============================================end of the code===========================
    #===========================new code for the funnel and user wise===================
    elif result_type == "source_wise_funnel":
        logger.info("Rendering source-wise funnel")
        funnel_data = result.get("funnel_data", {})
        st.subheader(f"Source-Wise Funnel Analysis{title_suffix}")
        st.info(f"Found {len(filtered_data)} leads matching the criteria.")

        if not funnel_data:
            st.warning("No funnel data available for any source.")
            return

        # Prepare data for a single table
        metrics_list = []
        sources = list(funnel_data.keys())
        if sources:
            all_metrics = list(funnel_data[sources[0]].keys())
            for metric in all_metrics:
                row = {"Metric": metric}
                for source in sources:
                    row[source] = funnel_data[source][metric]
                metrics_list.append(row)

            funnel_df = pd.DataFrame(metrics_list)
            for source in sources:
                funnel_df[source] = pd.to_numeric(funnel_df[source], errors='ignore')
            funnel_df = funnel_df.round(2)

            st.dataframe(funnel_df, use_container_width=True, hide_index=True)

        if st.button("Show Data", key=f"source_funnel_data_{result_type}_{selected_quarter}"):
            st.write(f"Filtered {object_type.capitalize()} Data")
            display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
            st.dataframe(display_data, use_container_width=True, hide_index=True)

    elif result_type == "user_wise_funnel":
        logger.info("Rendering user-wise funnel")
        
        funnel_data = result.get("funnel_data", {})
        st.subheader(f"User-Wise Funnel Analysis{title_suffix}")
        st.info(f"Found {len(filtered_data)} leads matching the criteria.")

        if not funnel_data:
            st.warning("No funnel data available for any user.")
            return

        # Prepare data for a single table
        metrics_list = []
        users = list(funnel_data.keys())
        if users:
            all_metrics = list(funnel_data[users[0]].keys())
            for metric in all_metrics:
                row = {"Metric": metric}
                for user in users:
                    row[user] = funnel_data[user][metric]
                metrics_list.append(row)

            funnel_df = pd.DataFrame(metrics_list)
            for user in users:
                funnel_df[user] = pd.to_numeric(funnel_df[user], errors='ignore')
            funnel_df = funnel_df.round(2)

            st.dataframe(funnel_df, use_container_width=True, hide_index=True)

        if st.button("Show Data", key=f"user_funnel_data_{result_type}_{selected_quarter}"):
            st.write(f"Filtered {object_type.capitalize()} Data")
            display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
            st.dataframe(display_data, use_container_width=True, hide_index=True)
    #===================================new code of the user wise follow up===========================
    elif result_type == "user_wise_follow_up":
        logger.info("Rendering user-wise follow-up")
        
        follow_up_data = result.get("follow_up_data", {})
        st.subheader(f"User-Wise Follow-Up Analysis{title_suffix}")
        st.info(f"Found {len(filtered_data)} tasks matching the criteria.")

        if not follow_up_data:
            st.warning("No follow-up data available for any user.")
            return

        # Prepare data for a single table: User | Subject | Count
        table_data = []
        users = list(follow_up_data.keys())
        if users:
            for user in users:
                for item in follow_up_data[user]:
                    table_data.append({
                        "User": user,
                        "Subject": item["Subject"],
                        "Count": item["Count"]
                    })

            # Create DataFrame for display
            follow_up_df = pd.DataFrame(table_data)
            follow_up_df["Count"] = pd.to_numeric(follow_up_df["Count"], errors='ignore')
            follow_up_df = follow_up_df.round(2)

            st.dataframe(follow_up_df, use_container_width=True, hide_index=True)

        if st.button("Show Data", key=f"user_follow_up_data_{result_type}_{selected_quarter}"):
            st.write(f"Filtered {object_type.capitalize()} Data")
            display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
            st.dataframe(display_data, use_container_width=True, hide_index=True)
    
    #============================== project wise follow up==================================
    elif result_type == "project_wise_follow_up":
        logger.info("Rendering project-wise follow-up")
        
        follow_up_data = result.get("follow_up_data", {})
        st.subheader(f"Project-Wise Follow-Up Analysis{title_suffix}")
        st.info(f"Found {len(filtered_data)} tasks matching the criteria.")

        if not follow_up_data:
            st.warning("No follow-up data available for any project.")
            return

        # Prepare data for a single table: Project | Subject | Count
        table_data = []
        projects = list(follow_up_data.keys())
        if projects:
            for project in projects:
                for item in follow_up_data[project]:
                    table_data.append({
                        "Project": project,
                        "Subject": item["Subject"],
                        "Count": item["Count"]
                    })

            # Create DataFrame for display
            follow_up_df = pd.DataFrame(table_data)
            follow_up_df["Count"] = pd.to_numeric(follow_up_df["Count"], errors='ignore')
            follow_up_df = follow_up_df.round(2)

            st.dataframe(follow_up_df, use_container_width=True, hide_index=True)

        if st.button("Show Data", key=f"project_follow_up_data_{result_type}_{selected_quarter}"):
            st.write(f"Filtered {object_type.capitalize()} Data")
            display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
            st.dataframe(display_data, use_container_width=True, hide_index=True)
    #=======================================  code  of the product and source  wise follow up=================
    elif result_type == "product_wise_follow_up":
        logger.info("Rendering product-wise follow-up")
        
        follow_up_data = result.get("follow_up_data", {})
        st.subheader(f"Product-Wise Follow-Up Analysis{title_suffix}")
        st.info(f"Found {len(filtered_data)} tasks matching the criteria.")

        if not follow_up_data:
            st.warning("No follow-up data available for any product.")
            return

        # Prepare data for a single table: Product | Subject | Count
        table_data = []
        products = list(follow_up_data.keys())
        if products:
            for product in products:
                for item in follow_up_data[product]:
                    table_data.append({
                        "Product": product,
                        "Subject": item["Subject"],
                        "Count": item["Count"]
                    })

            # Create DataFrame for display
            follow_up_df = pd.DataFrame(table_data)
            follow_up_df["Count"] = pd.to_numeric(follow_up_df["Count"], errors='ignore')
            follow_up_df = follow_up_df.round(2)

            st.dataframe(follow_up_df, use_container_width=True, hide_index=True)

        if st.button("Show Data", key=f"product_follow_up_data_{result_type}_{selected_quarter}"):
            st.write(f"Filtered {object_type.capitalize()} Data")
            display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
            st.dataframe(display_data, use_container_width=True, hide_index=True)

    elif result_type == "source_wise_follow_up":
        logger.info("Rendering source-wise follow-up")
        
        follow_up_data = result.get("follow_up_data", {})
        st.subheader(f"Source-Wise Follow-Up Analysis{title_suffix}")
        st.info(f"Found {len(filtered_data)} tasks matching the criteria.")

        if not follow_up_data:
            st.warning("No follow-up data available for any source.")
            return

        # Prepare data for a single table: Source | Subject | Count
        table_data = []
        sources = list(follow_up_data.keys())
        if sources:
            for source in sources:
                for item in follow_up_data[source]:
                    table_data.append({
                        "Source": source,
                        "Subject": item["Subject"],
                        "Count": item["Count"]
                    })

            # Create DataFrame for display
            follow_up_df = pd.DataFrame(table_data)
            follow_up_df["Count"] = pd.to_numeric(follow_up_df["Count"], errors='ignore')
            follow_up_df = follow_up_df.round(2)

            st.dataframe(follow_up_df, use_container_width=True, hide_index=True)

        if st.button("Show Data", key=f"source_follow_up_data_{result_type}_{selected_quarter}"):
            st.write(f"Filtered {object_type.capitalize()} Data")
            display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
            st.dataframe(display_data, use_container_width=True, hide_index=True)
    #==========================open lead not follow up=====================
    
    elif result_type == "open_lead_not_follow_up":
        logger.info("Rendering open lead without follow-up analysis")
        
        open_lead_data = result.get("open_lead_data", {})
        st.subheader(f"Open Lead Without Follow-Up Analysis{title_suffix}")
        st.info(f"Found {len(filtered_data)} tasks matching the criteria.")

        if not open_lead_data:
            st.warning("No open leads without follow-up found.")
            return

        # Prepare data for a single table: Customer | Subject | Count
        table_data = []
        customers = list(open_lead_data.keys())
        if customers:
            for customer in customers:
                for item in open_lead_data[customer]:
                    table_data.append({
                        "Customer": customer,
                        "Subject": item["Subject"],
                        "Count": item["Count"]
                    })

            # Create DataFrame for display
            open_lead_df = pd.DataFrame(table_data)
            open_lead_df["Count"] = pd.to_numeric(open_lead_df["Count"], errors='ignore')
            open_lead_df = open_lead_df.round(2)

            st.dataframe(open_lead_df, use_container_width=True, hide_index=True)

        if st.button("Show Data", key=f"open_lead_follow_up_data_{result_type}_{selected_quarter}"):
            st.write(f"Filtered {object_type.capitalize()} Data")
            display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
            st.dataframe(display_data, use_container_width=True, hide_index=True)

   #===========================open lead not follow up ===========================
    
    #=========================== end of code===================================
    #================================start crm team member===================
    elif result_type == "crm_team_member":
        logger.info("Rendering crm-team member")
        
        funnel_data = result.get("funnel_data", {})
        st.subheader(f"crm-team member Analysis{title_suffix}")
        st.info(f"Found {len(filtered_data)} leads matching the criteria.")

        if not funnel_data:
            st.warning("No funnel data available for any user.")
            return

        # Prepare data for a single table
        metrics_list = []
        users = list(funnel_data.keys())
        if users:
            all_metrics = list(funnel_data[users[0]].keys())
            for metric in all_metrics:
                row = {"Metric": metric}
                for user in users:
                    row[user] = funnel_data[user][metric]
                metrics_list.append(row)

            funnel_df = pd.DataFrame(metrics_list)
            for user in users:
                funnel_df[user] = pd.to_numeric(funnel_df[user], errors='ignore')
            funnel_df = funnel_df.round(2)

            st.dataframe(funnel_df, use_container_width=True, hide_index=True)

        if st.button("Show Data", key=f"crm_team_data_{result_type}_{selected_quarter}"):
            st.write(f"Filtered {object_type.capitalize()} Data")
            display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
            st.dataframe(display_data, use_container_width=True, hide_index=True)
    #================================end of code================================
    
    elif result_type == "product_wise_lead":
        logger.info("Rendering product-wise lead summary")
        funnel_data = result.get("funnel_data", pd.DataFrame())
        st.subheader(f"Product-Wise Lead Summary{title_suffix}")
        st.info(f"Found {len(filtered_data)} leads matching the criteria.")

        if st.button("Show Data", key=f"product_lead_data_{result_type}_{selected_quarter}"):
            st.write(f"Filtered {object_type.capitalize()} Data")
            display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
            st.dataframe(display_data, use_container_width=True, hide_index=True)

        if not funnel_data.empty:
            st.subheader("Product-Wise Lead Counts")
            st.info("Counts grouped by Product Category")
            # Ensure only 'Project_Category__c' and 'Count' are displayed
            funnel_data = funnel_data[['Project_Category__c', 'Count']].sort_values(by="Count", ascending=False)
            # Rename 'Project_Category__c' to 'product' for clarity
            funnel_data = funnel_data.rename(columns={"Project_Category__c": "product"})
            st.dataframe(funnel_data, use_container_width=True, hide_index=True)
   
    elif result_type == "quarterly_distribution":
        logger.info("Rendering quarterly distribution")
        fields = result.get("fields", [])
        quarterly_data = result.get("data", {})
        logger.info(f"Quarterly data: {quarterly_data}")
        logger.info(f"Quarterly data keys: {list(quarterly_data.keys())}")
        for key in quarterly_data.keys():
            logger.info(f"Quarterly data key: '{key}' (length: {len(key)})")
            logger.info(f"Quarterly data key bytes: {list(key.encode('utf-8'))}")
        if not quarterly_data:
            st.info(f"No {object_type} data found.")
            return
        st.subheader(f"Quarterly {object_type.capitalize()} Results{title_suffix}")
        field = fields[0] if fields else None
        field_display = FIELD_DISPLAY_NAMES.get(field, field) if field else "Field"

        if not filtered_data.empty:
            st.info(f"Found {len(filtered_data)} rows.")
            show_data = st.button("Show Data", key=f"show_data_quarterly_{result_type}_{normalized_quarter}")
            if show_data:
                st.write(f"Filtered {object_type.capitalize()} Data")
                display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
                st.dataframe(display_data, use_container_width=True, hide_index=True)

        normalized_quarterly_data = {k.strip(): v for k, v in quarterly_data.items()}
        logger.info(f"Normalized quarterly data keys: {list(normalized_quarterly_data.keys())}")
        for key in normalized_quarterly_data.keys():
            logger.info(f"Normalized key: '{key}' (length: {len(key)})")
            logger.info(f"Normalized key bytes: {list(key.encode('utf-8'))}")

        dist = None
        if normalized_quarter in normalized_quarterly_data:
            dist = normalized_quarterly_data[normalized_quarter]
            logger.info(f"Found exact match for quarter: {normalized_quarter}")
        else:
            for key in normalized_quarterly_data.keys():
                if key == normalized_quarter:
                    dist = normalized_quarterly_data[key]
                    logger.info(f"Found matching key after strict comparison: '{key}'")
                    break
                if list(key.encode('utf-8')) == list(normalized_quarter.encode('utf-8')):
                    dist = normalized_quarterly_data[key]
                    logger.info(f"Found matching key after byte-level comparison: '{key}'")
                    break

        logger.info(f"Final distribution for {normalized_quarter}: {dist}")
        if not dist:
            if quarterly_data:
                for key, value in quarterly_data.items():
                    if "Q4" in key:
                        dist = value
                        logger.info(f"Forcing display using key: '{key}' with data: {dist}")
                        break
            if not dist:
                st.info(f"No data found for {normalized_quarter}.")
                return

        quarter_df = pd.DataFrame.from_dict(dist, orient='index', columns=['Count']).reset_index()
        if object_type == "lead" and field == "Customer_Feedback__c":
            quarter_df['index'] = quarter_df['index'].map({
                'Interested': 'Interested',
                'Not Interested': 'Not Interested'
            })
        quarter_df.columns = [f"{field_display}", "Count"]
        quarter_df = quarter_df.sort_values(by="Count", ascending=False)
        st.dataframe(quarter_df, use_container_width=True, hide_index=True)

    elif result_type == "source_wise_lead":
        logger.info("Rendering source-wise funnel")
        funnel_data = result.get("funnel_data", pd.DataFrame())
        st.subheader(f"{object_type.capitalize()} Results{title_suffix}")
        st.info(f"Found {len(filtered_data)} rows.")

        if st.button("Show Data", key=f"source_funnel_data_{result_type}_{selected_quarter}"):
            st.write(f"Filtered {object_type.capitalize()} Data")
            display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
            st.dataframe(display_data, use_container_width=True, hide_index=True)

        if not funnel_data.empty:
            st.subheader("Source-Wise Lead")
            st.info("Counts grouped by Source")
            funnel_data = funnel_data.sort_values(by="Count", ascending=False)
            st.dataframe(funnel_data.rename(columns=FIELD_DISPLAY_NAMES), use_container_width=True, hide_index=True)

    elif result_type == "table":
        logger.info("Rendering table result")
        data = result.get("data", [])
        data_df = pd.DataFrame(data)
        if data_df.empty:
            st.info(f"No {object_type} data found.")
            return
        st.subheader(f"{object_type.capitalize()} Results{title_suffix}")
        st.info(f"Found {len(data_df)} rows.")

        if st.button("Show Data", key=f"table_data_{result_type}_{selected_quarter}"):
            st.write(f"Filtered {object_type.capitalize()} Data")
            display_data, display_cols = prepare_filtered_display_data(data_df, analysis_plan)
            st.dataframe(display_data, use_container_width=True, hide_index=True)


#============================= end of code=======================================

    elif result_type == "distribution":
        logger.info("Rendering distribution result")
        data = result.get("data", {})
        st.subheader(f"Distribution Results{title_suffix}")

        if not filtered_data.empty:
            st.info(f"Found {len(filtered_data)} rows.")
            if st.button("Show Data", key=f"dist_data_{result_type}_{selected_quarter}"):
                st.write(f"Filtered {object_type.capitalize()} Data")
                display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
                st.dataframe(display_data, use_container_width=True, hide_index=True)

        for field, dist in data.items():
            st.write(f"Distribution of {FIELD_DISPLAY_NAMES.get(field, field)}")
            dist_df = pd.DataFrame.from_dict(dist["counts"], orient='index', columns=['Count']).reset_index()
            dist_df.columns = [f"{FIELD_DISPLAY_NAMES.get(field, field)}", "Count"]
            dist_df["Percentage"] = pd.DataFrame.from_dict(dist["percentages"], orient='index').values
            dist_df = dist_df.sort_values(by="Count", ascending=False)
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

    elif result_type == "percentage":
        logger.info("Rendering percentage result")
        st.subheader(f"Percentage Analysis{title_suffix}")
        st.metric(result.get("label", "Percentage"), f"{result.get('value', 0)}%")

    elif result_type == "info":
        logger.info("Rendering info message")
        st.info(result.get("message", "No specific message provided"))
        return

    elif result_type == "error":
        logger.error("Rendering error message")
        st.error(result.get("message", "An error occurred"))
        return
    
    elif result_type == "user_meeting_summary":
        logger.info("Rendering user-wise meeting summary")
        st.subheader(f"User-Wise Meeting Done Summary{title_suffix}")
        df = pd.DataFrame(result["data"])
        if not df.empty:
            st.dataframe(df.rename(columns={"Name": "User", "Department": "Department", "Meeting_Done_Count": "Completed Meetings"}), use_container_width=True, hide_index=True)
            total_meetings = result.get("total", 0)  # Safely get total, default to 0 if not present
            st.info(f"Total completed meetings: {total_meetings}")
        else:
            st.warning("No completed meeting data found for the selected criteria.")
            
    elif result_type == "dept_user_meeting_summary":
        logger.info("Rendering department-wise user meeting summary")
        st.subheader(f"Department-Wise User Meeting Done Summary{title_suffix}")
        df = pd.DataFrame(result["data"])
        if not df.empty:
            column_mapping = {col: col_display_name.get(col, col) for col in result["columns"]}
            st.dataframe(df.rename(columns=column_mapping), use_container_width=True, hide_index=True)
            total_meetings = result.get("total", 0)  # Safely get total, default to 0 if not present
            st.info(f"Total completed meetings: {total_meetings}")
        else:
            st.warning("No completed meeting data found for the selected criteria.")
            
    elif result_type == "user_sales_summary":
        logger.info("Rendering user-wise sales summary")
        st.subheader(f"User-Wise Sales Order Summary{title_suffix}")
        df = pd.DataFrame(result["data"])
        if not df.empty:
            st.dataframe(df.rename(columns={"Name": "User", "Sales_Order_Count": "Sales Orders"}), use_container_width=True, hide_index=True)
            total_sales = result.get("total", 0)
            st.info(f"Total sales orders: {total_sales}")
        else:
            st.warning("No sales order data found for the selected criteria.")

    # Show Graph button for all applicable result types
    if result_type not in ["info", "error"]:
        show_graph = st.button("Show Graph", key=f"show_graph_{result_type}_{selected_quarter}")
        if show_graph:
            st.subheader(f"Graph{title_suffix}")
            display_data, display_cols = prepare_filtered_display_data(filtered_data, analysis_plan)
            relevant_graph_fields = [f for f in display_cols if f in graph_data]
            if result_type == "quarterly_distribution":
                render_graph(graph_data.get(normalized_quarter, {}), relevant_graph_fields, title_suffix)

            # For opportunity_vs_lead, explicitly include "Opportunity vs Lead"
            elif result_type == "opportunity_vs_lead":
                relevant_graph_fields = ["Opportunity vs Lead"]
                render_graph(graph_data, relevant_graph_fields, title_suffix)
            elif result_type == "conversion_funnel":
                # For conversion funnel, we pass quarterly_data to align funnel stages with the table
                quarterly_data_for_graph = result.get("quarterly_data", {}).get(selected_quarter, {})
                render_graph(graph_data, ["Funnel Stages"], title_suffix, quarterly_data=quarterly_data_for_graph)
            elif result_type == "product_wise_funnel":
                render_graph(graph_data, ["Product Funnel Stages"], title_suffix, quarterly_data=funnel_data)
            
            elif result_type == "user_sales_summary":
                relevant_graph_fields = ["User_Sales"]
                render_graph(graph_data, relevant_graph_fields, title_suffix)
           
            elif result_type == "dept_user_meeting_summary":
                relevant_graph_fields = ["Dept_Meeting_Done"]
                render_graph(graph_data, relevant_graph_fields, title_suffix)
            elif result_type == "user_meeting_summary":
                relevant_graph_fields = ["User_Meeting_Done"]
                render_graph(graph_data, relevant_graph_fields, title_suffix)
            else:
                render_graph(graph_data, relevant_graph_fields, title_suffix)

        # Add Export to CSV option for applicable result types
        if result_type in ["table", "distribution", "quarterly_distribution", "source_wise_lead", "product_wise_lead", "conversion_funnel", "product_wise_funnel"]:
            if not filtered_data.empty:
                export_key = f"export_data_{result_type}_{selected_quarter}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if st.button("Export Data to CSV", key=export_key):
                    file_name = f"{result_type}_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    filtered_data.to_csv(file_name, index=False)
                    st.success(f"Data exported to {file_name}")

        # Add a separator for better UI separation
        st.markdown("---")

if __name__ == "__main__":
    st.title("Analysis Dashboard")
    # Add a button to clear Streamlit cache
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared successfully!")
    user_question = st.text_input("Enter your query:", "product-wise sale")
    if st.button("Analyze"):
        # Sample data for testing
        sample_data = {
            "CreatedDate": [
                "2024-05-15T10:00:00Z",
                "2024-08-20T12:00:00Z",
                "2024-11-10T08:00:00Z",
                "2025-02-15T09:00:00Z"
            ],
            "Project_Category__c": [
                "ARANYAM VALLEY",
                "HARMONY GREENS",
                "DREAM HOMES",
                "ARANYAM VALLEY"
            ],
            "Customer_Feedback__c": [
                "Interested",
                "Not Interested",
                "Interested",
                "Not Interested"
            ],
            "Disqualification_Reason__c": [
                "Budget Issue",
                "Not Interested",
                "Budget Issue",
                "Location Issue"
            ],
            "Status": [
                "Qualified",
                "Unqualified",
                "Qualified",
                "New"
            ],
            "Is_Appointment_Booked__c": [
                True,
                False,
                True,
                False
            ],
            "LeadSource": [
                "Facebook",
                "Google",
                "Website",
                "Facebook"
            ]
        }
        leads_df = pd.DataFrame(sample_data)
        users_df = pd.DataFrame()
        cases_df = pd.DataFrame()
        events_df = pd.DataFrame({ "Status": ["Completed"],  # Added sample task data to test Meeting Done
            "CreatedDate": ["2025-02-15T10:00:00Z", "2025-02-15T11:00:00Z"]})
        opportunities_df = pd.DataFrame({
            "Sales_Order_Number__c": [123, 456, 789],
            "Project_Category__c": ["VERIDIA", "ELIGO", "EDEN", "WAVE GARDEN"],
            "CreatedDate": ["2025-02-15T10:00:00Z", "2025-02-15T11:00:00Z", "2025-02-15T12:00:00Z", "2025-02-15T13:00:00Z"]
        })
        task_df = pd.DataFrame({
            "Status": ["Completed", "Open"],  # Added sample task data to test Meeting Done
            "CreatedDate": ["2025-02-15T10:00:00Z", "2025-02-15T11:00:00Z"]
        })

        
        analysis_plan = {
            "analysis_type": "customer_feedback_summary" if "customer feedback" in user_question.lower() or "feedback summary" in user_question.lower() else "distribution",
            "object_type": "lead",
            "fields": ["Customer_Feedback__c"] if "customer feedback" in user_question.lower() or "feedback summary" in user_question.lower() else ["Project_Category__c"],
            "quarter": "Q1 - Q4",
            "filters": {}
        }
        result = execute_analysis(analysis_plan, leads_df, users_df, cases_df, events_df, opportunities_df, task_df, user_question)
        display_analysis_result(result, analysis_plan, user_question)


#===============================new code====================
import os
import requests
import json
import re
import pandas as pd
import aiohttp
import asyncio
from urllib.parse import quote
import datetime
from pytz import timezone
from dotenv import load_dotenv
import logging
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import uvicorn

# === Configuration Section ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# FastAPI App
app = FastAPI()

# Authentication
API_KEY = os.getenv("ORCHESTRATE_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key



# Request Model
class AnalysisRequest(BaseModel):
    question: str
  

# Response Model
class AnalysisResponse(BaseModel):
    analysis_type: str
    insight: str
    data: dict
    visualization: dict = None
    success: bool = True
    error: str = None

# === Core Salesforce Functions ===
async def get_access_token():
    """Authenticate with Salesforce and get access token"""
    if not all([client_id, client_secret, username, password, login_url]):
        raise ValueError("Missing Salesforce credentials")
    
    payload = {
        'grant_type': 'password',
        'client_id': client_id,
        'client_secret': client_secret,
        'username': username,
        'password': password
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(login_url, data=payload, timeout=30) as res:
                res.raise_for_status()
                token_data = await res.json()
                return token_data['access_token']
    except Exception as e:
        logger.error(f"Salesforce auth failed: {e}")
        raise

async def load_salesforce_data_async():
    """Load all Salesforce data asynchronously"""
    try:
        access_token = await get_access_token()
        async with aiohttp.ClientSession() as session:
            headers = {'Authorization': f'Bearer {access_token}'}
            
            # Execute all queries concurrently
            leads, users, cases, events, opportunities, tasks = await asyncio.gather(
                load_object_data(session, access_token, SF_LEADS_URL, 
                               {'minimal': get_minimal_lead_fields(), 
                                'standard': get_standard_lead_fields(), 
                                'extended': get_extended_lead_fields()},
                               "Lead"),
                load_object_data(session, access_token, SF_USERS_URL, 
                               {'minimal': get_minimal_user_fields(), 
                                'standard': get_standard_user_fields(), 
                                'extended': get_extended_user_fields()},
                               "User", date_filter=False),
                load_object_data(session, access_token, SF_CASES_URL, 
                               {'minimal': get_minimal_case_fields(), 
                                'standard': get_standard_case_fields(), 
                                'extended': get_extended_case_fields()},
                               "Case"),
                load_object_data(session, access_token, SF_EVENTS_URL, 
                               {'minimal': get_minimal_event_fields(), 
                                'standard': get_standard_event_fields(), 
                                'extended': get_extended_event_fields()},
                               "Event"),
                load_object_data(session, access_token, SF_OPPORTUNITIES_URL, 
                               {'minimal': get_minimal_opportunity_fields(), 
                                'standard': get_standard_opportunity_fields(), 
                                'extended': get_extended_opportunity_fields()},
                               "Opportunity"),
                load_object_data(session, access_token, SF_TASKS_URL, 
                               {'minimal': get_minimal_task_fields(), 
                                'standard': get_standard_task_fields(), 
                                'extended': get_extended_task_fields()},
                               "Task")
            )
            
            return leads, users, cases, events, opportunities, tasks
            
    except Exception as e:
        logger.error(f"Error loading Salesforce data: {str(e)}")
        raise

def load_salesforce_data():
    """Synchronous wrapper for async data loading"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(load_salesforce_data_async())
    except Exception as e:
        logger.error(f"Error in event loop: {str(e)}")
        raise


async def analyze_question(question: str):
    """Main analysis function"""
    try:
        # Load data asynchronously
        leads, users, cases, events, opportunities, tasks = await load_salesforce_data_async()
        
        # Create context for AI
        data_context = create_data_context(leads, users, cases, events, opportunities, tasks)
        
        # Get analysis plan from AI
        analysis_plan = query_watsonx_ai(question, data_context, leads, cases, events, users, opportunities, tasks)
        
        # Execute analysis
        result = execute_analysis(analysis_plan, leads, users, cases, events, opportunities, tasks, question)
        
        # Format for Orchestrate
        return format_for_orchestrate(result)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return AnalysisResponse(
            success=False,
            error=str(e),
            analysis_type="error",
            insight="Failed to analyze question",
            data={}
        )



def format_for_orchestrate(result: dict) -> AnalysisResponse:
    try:
        data = result.get("data", {})
        if isinstance(data, list) and data:
            data = {item.get("Department", "Unknown"): item.get("Meeting_Done_Count", 0) for item in data}
        response = AnalysisResponse(
            analysis_type=result.get("type", "unknown"),
            insight=result.get("explanation", "Analysis completed"),
            data=data,
            visualization=result.get("graph_data", {})
        )
        return response
    except Exception as e:
        logger.error(f"Formatting failed: {str(e)}")
        return AnalysisResponse(
            success=False,
            error=str(e),
            analysis_type="error",
            insight="Failed to format results",
            data={}
        )

# === API Endpoints ===
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_salesforce(
    request: AnalysisRequest, 
    api_key: str = Depends(get_api_key)
):
    """Main endpoint for Watsonx Orchestrate integration"""
    try:
        logger.info(f"Analyzing question: {request.question}")
        return await analyze_question(request.question)  # Note the 'await' here
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



# === Main Execution ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)