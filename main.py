

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
# === Salesforce Utilities Section ===
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
    start_date = "2024-04-01T00:00:00Z"
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

# def load_salesforce_data():
#     try:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         return loop.run_until_complete(load_salesforce_data_async())
#     except Exception as e:
#         error_msg = f"Error in event loop: {str(e)}"
#         logger.error(error_msg)
#         return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), error_msg

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
            "available_event_fields": list(events_df.columns) if not events_df.empty else [],
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
            sample_values = task_df[col].dropna().unique()[:5].tolist()
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
        #===============================new code for the user=======================
        # Detect user-wise sales queries
        if any(keyword in user_question.lower() for keyword in ["sale by user", "user wise sale", "user-wise sale", "sales by user"]):
            return {
                "analysis_type": "user_sales_summary",
                "object_type": "opportunity",
                "fields": ["OwnerId", "Sale_Order_Number__c"],
                
                "explanation": "Show count of closed-won opportunities grouped by user"
            }
            
        if "user wise meeting done" in user_question.lower():
            return {
                "analysis_type": "user_meeting_summary",
                "object_type": "event",
                "fields": ["OwnerId", "Appointment_Status__c"],
                "filters": {"Appointment_Status__c": "Completed"},
                "explanation": "Show count of completed meetings grouped by user"
            }
        if "department wise meeting done" in user_question.lower():
            return {
                "analysis_type": "dept_user_meeting_summary",
                "object_type": "event",
                "fields": ["OwnerId", "Appointment_Status__c", "Department"],
                "filters": {"Appointment_Status__c": "Completed"},
                "explanation": "Show count of completed meetings grouped by user and department"
            }
            
        if "total meeting done" in user_question.lower():
            return {
                "analysis_type": "count",
                "object_type": "event",
                "fields": ["Appointment_Status__c"],
                "filters": {"Appointment_Status__c": "Completed"},
                "explanation": "Show total count of completed meetings"
            }
        #=================== end of code for new user========================================
        
        # Updated to handle both singular and plural forms
        if "disqualification reason" in user_question.lower() or "disqualification reasons" in user_question.lower():
            return {
                "analysis_type": "disqualification_summary",
                "object_type": "lead",
                "field": "Disqualification_Reason__c",
                "filters": {},
                "explanation": "Show disqualification reasons with count and percentage"
            }
            
        if "junk reason" in user_question.lower():
            return {
                "analysis_type": "junk_reason_summary",
                "object_type": "lead",
                "field": "Junk_Reason__c",
                "filters": {},
                "explanation": "Show junk reasons with count and percentage"
            }
            
        if any(keyword in user_question.lower() for keyword in ["disqualification"]) and any(pct in user_question.lower() for pct in ["%", "percent", "percentage"]):
            return {
                "analysis_type": "percentage",
                "object_type": "lead",
                "fields": ["Customer_Feedback__c"],  # Add the field being filtered
                "filters": {"Customer_Feedback__c": "Not Interested"},
                "explanation": "Calculate percentage of disqualification leads where Customer_Feedback__c is 'Not Interested'"
            }

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
            "follow up": "Follow_Up_Date_Time__c",
            "location": "Preferred_Location__c",
            "crm team member": "OwnerId",
            "lead to sale ratio": "Status",
            "time between contact and conversion": "CreatedDate",
            "follow-up consistency": "Follow_UP_Remarks__c",
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
            # Project categories
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
            "total appointments":"Appointment_status__c",
            "user wise meeting done": ["OwnerId", "Appointment_Status__c"]
        }

        opportunity_keyword_mappings = {
            "qualified opportunity":  "Sales_Team_Feedback__c",
            "disqualified opportunity": "Sales_Team_Feedback__c",
            "amount": "Amount",
            "close date": "CloseDate",
            "opportunity type": "Opportunity_Type__c",
            "new business": "Opportunity_Type__c",
            "renewal": "Opportunity_Type__c",
            "upsell": "Opportunity_Type__c",
            "cross-sell": "Opportunity_Type__c",
            "total sale": "Sales_Order_Number__c",
            # Add new mappings for product-wise sales
            # Add new mappings for source-wise sales
            "source-wise sale": "LeadSource",
            "source with sale": "LeadSource",
            # Add new mappings for lead source subcategory
            "lead source subcategory with sale": "Lead_Source_Sub_Category__c",
            "subcategory with sale": "Lead_Source_Sub_Category__c",
            "product-wise sales": "Project_Category__c",
            "products with sales": "Project_Category__c",
            "product sale": "Project_Category__c",
            "sale": "Sales_Order_Number__c",
            "project-wise sales": "Project__c",
            "project with sale":  "Project__c",
            "project sale": "Project__c",
            "sales by user": "OwnerId",
            "user-wise sale": "OwnerId"
            
        }

        task_keyword_mappings = {
            "task status": "Status",
            "follow up status": "Follow_Up_Status__c",
            "task feedback": "Customer_Feedback__c",
            "sales feedback": "Sales_Team_Feedback__c",
            "transfer status": "Transfer_Status__c",
            "task subject": "Subject",
            "completed task": "Status",
            "open task": "Status",
            "pending follow-up": "Follow_Up_Status__c",
            "no follow-up": "Follow_Up_Status__c"
            
        }

        # Detect quarter from user question
        quarter_mapping = {
            r'\b(q1|quarter\s*1|first\s*quarter)\b': 'Q1 2024-25',
            r'\b(q2|quarter\s*2|second\s*quarter)\b': 'Q2 2024-25',
            r'\b(q3|quarter\s*3|third\s*quarter)\b': 'Q3 2024-25',
            r'\b(q4|quarter\s*4|fourth\s*quarter)\b': 'Q4 2024-25',
        }
        selected_quarter = None
        question_lower = user_question.lower()
        for pattern, quarter in quarter_mapping.items():
            if re.search(pattern, question_lower, re.IGNORECASE):
                selected_quarter = quarter
                logger.info(f"Detected quarter: {selected_quarter} for query: {user_question}")
                break
        # Default to Q4 2024-25 for quarterly_distribution if no quarter specified
        if "quarter" in question_lower and not selected_quarter:
            selected_quarter = "Q4 2024-25"
            logger.info(f"No specific quarter detected, defaulting to {selected_quarter}")

        system_prompt = f"""
You are an intelligent Salesforce analytics assistant. Your task is to convert user questions into a JSON-based analysis plan for lead, case, event, opportunity, or task data.

Available lead fields: {sample_lead_fields}
Available lead fields: {sample_user_fields}
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
- Use keyword-to-column mappings to select the correct field (e.g., "disqualification reason" → `Disqualification_Reason__c`).
- For terms like "2BHK", "3BHK", filter `Property_Size__c` (e.g., `Property_Size__c: ["2BHK", "3BHK"]`).
- For "residential" or "commercial", filter `Property_Type__c` (e.g., `Property_Type__c: "Residential"`).
- For project categories (e.g., "ARANYAM VALLEY"), filter `Project_Category__c` (e.g., `Project_Category__c: "ARANYAM VALLEY"`).
- For "interested", filter `Customer_Feedback__c = "Interested"`.
- For "qualified opportunity", filter `Sales_Team_Feedback__c = "Qualified"` for opportunities.
- For "disqualified opportunity", filter `Sales_Team_Feedback__c = "Disqualified"` or `Sales_Team_Feedback__c = "Not Interested"` for opportunities (use the value that matches your data).
- For "hot lead", "cold lead", "warm lead", filter `Rating` (e.g., `Rating: "Hot"`).
- For "qualified", filter `Customer_Feedback__c = "Interested"`.
- For "disqualified", "disqualification", or "unqualified", filter `Customer_Feedback__c = "Not Interested"`.

- For "total sale", filter `Sales_Order_Number__c` where it is not null (i.e., `Sales_Order_Number__c: {{"$ne": null}}`) for opportunities to count completed sales.
- For "sale", filter `Sales_Order_Number__c` where it is not null (i.e., `Sales_Order_Number__c: {{"$ne": null}}`) for opportunities to count completed sales.
- For "product-wise sales" or "products with sales", set `analysis_type` to `distribution`, `object_type` to `opportunity`, `field` to `Project_Category__c`, and filter `Sales_Order_Number__c: {{"$ne": null}}` to include only opportunities with completed sales. Group by `Project_Category__c` to show the count of sales per product.

- For "project-wise sale", "project with sale", or "project sale", set `analysis_type` to `distribution`, `object_type` to `opportunity`, `field` to `Project__c`, and filter `Sales_Order_Number__c: {{"$ne": null}}` to include only opportunities with completed sales. Group by `Project__c` to show the count of sales per project.
- For "source-wise sale" or "source with sale", set `analysis_type` to `distribution`, `object_type` to `opportunity`, `field` to `LeadSource`, and filter `Sales_Order_Number__c: {{"$ne": null}}` to include only opportunities with completed sales. Group by `LeadSource` to show the count of sales per source.
- For "lead source subcategory with sale" or "subcategory with sale", set `analysis_type` to `distribution`, `object_type` to `opportunity`, `field` to `Lead_Source_Sub_Category__c`, and filter `Sales_Order_Number__c: {{"$ne": null}}` to include only opportunities with completed sales. Group by `Lead_Source_Sub_Category__c` to show the count of sales per subcategory.

- For "open lead", filter `Customer_Feedback__c` in `["Discussion Pending", null]` (i.e., `Customer_Feedback__c: {{"$in": ["Discussion Pending", null]}}`).
- For " lead convert opportunity" or "lead versus opportunity" queries (including "how many", "breakdown", "show me", or "%"), set `analysis_type` to `opportunity_vs_lead` for counts or `opportunity_vs_lead_percentage` for percentages. Use `Customer_Feedback__c = Interested` for opportunities and count all `Id` for leads.
- Data is available from 2024-04-01T00:00:00Z to 2025-03-31T23:59:59Z. Adjust dates outside this range to the nearest valid date.
- For date-specific queries (e.g., "4 January 2024"), filter `CreatedDate` for that date.
- For "today", use 2025-06-13T00:00:00Z to 2025-06-13T23:59:59Z (UTC).
- For "last week" or "last month", calculate relative to 2025-06-13T00:00:00Z (UTC).
- For Hinglish like "2025 ", filter `CreatedDate` for that year.
- For "sale by user" or "user-wise sale", set `analysis_type` to `user_sales_summary`, `object_type` to `opportunity`, and join `opportunities_df` with `users_df` on `OwnerId` (opportunities) to `Id` (users).
- For non-null filters, use `{{"$ne": null}}`.
- If the user mentions "task status", use the `Status` field for tasks.

- If the user mentions "Total Appointment",use the `Appointment_Status__c` is in ["Completed",""Scheduled","Cancelled","No show"] within the `conversion_funnel` analysis.
- If the user mentions "completed task", map to `Status` with value "Completed" for tasks.
- If the user mentions "pending follow-up", map to `Follow_Up_Status__c` with value "Pending" for tasks.
- If the user mentions "interested", map to `Customer_Feedback__c` with value "Interested" for leads or tasks.
- If the user mentions "not interested", map to `Customer_Feedback__c` with value "Not Interested" for leads or tasks.
- If the user mentions "meeting done", map to `Appointment_Status__c` with value "Completed" for events.
- If the user mentions "meeting booked", map to `Status` with value "Qualified" for leads.

- If the user mentions "user wise meeting done", set `analysis_type` to `user_meeting_summary`, `object_type` to `event`, and join `events_df` with `users_df` on `OwnerId` (events) to `Id` (users). Count events where `Appointment_Status__c = "Completed"`, grouped by user name.

## Quarter Detection:
- Detect quarters from keywords:
  - "Q1", "quarter 1", "first quarter" → "Q1 2024-25" (2024-04-01T00:00:00Z to 2024-06-30T23:59:59Z)
  - "Q2", "quarter 2", "second quarter" → "Q2 2024-25" (2024-07-01T00:00:00Z to 2024-09-30T23:59:59Z)
  - "Q3", "quarter 3", "third quarter" → "Q3 2024-25" (2024-10-01T00:00:00Z to 2024-12-31T23:59:59Z)
  - "Q4", "quarter 4", "fourth quarter" → "Q4 2024-25" (2025-01-01T00:00:00Z to 2025-03-31T23:59:59Z)
- For `quarterly_distribution`, include `quarter` in the response (e.g., `quarter: "Q1 2024-25"`).
- If no quarter is specified for `quarterly_distribution`, default to "Q1 - Q4".
- For `quarterly_distribution` or `opportunity_vs_lead`, include `quarter` in the response (e.g., `quarter: "Q1 2024-25"`).
- If no quarter is specified for `quarterly_distribution` or `opportunity_vs_lead`, default to "Q1 - Q4".

## Analysis Types:
- count: Count records.
- distribution: Frequency of values.
- filter: List records.
- recent: Recent records.
- top: Top values.
- percentage: Percentage of matching records.
- quarterly_distribution: Group by quarters.
- source_wise_funnel: Group by `LeadSource` and `Lead_Source_Sub_Category__c`.
- conversion_funnel: Compute funnel metrics (Total Leads, Valid Leads, SOL, Meeting Booked, etc.).
- opportunity_vs_lead: Compare count of leads (all `Id`) with opportunities (`Customer_Feedback__c = Interested`).
- opportunity_vs_lead_percentage: Calculate percentage of leads converted to opportunities (`Customer_Feedback__c = Interested` / total leads).
- user_meeting_summary: Count completed meetings (`Appointment_Status__c = "Completed"`) per user.
- dept_user_meeting_summary: Count completed meetings (`Appointment_Status__c = "Completed"`) per user and department.
- user_sales_summary: Count closed-won opportunities  per user, joining `opportunities_df` with `users_df` on `OwnerId` to `Id`.
## Lead Conversion Funnel:
For "lead conversion funnel" or "funnel analysis":
- `analysis_type`: "conversion_funnel"
- Fields: `["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c"]`
- Metrics:
  - Total Leads: All leads.
  - Valid Leads: `Customer_Feedback__c != "Junk"`.
  - SOL: `Status = "Qualified"`.
  - Meeting Booked: `Status = "Qualified"` and `Is_Appointment_Booked__c = True`.
  - Disqualified Leads: `Customer_Feedback__c = "Not Interested"`.
  - Open Leads: `Customer_Feedback__c` in `["Discussion Pending", None]`.
  - Total Appointment : `Appointment_Status__c` in `["Completed",""Scheduled","Cancelled","No show"]`.
  - Junk %: ((Total Leads - Valid Leads) / Total Leads) * 100.
  - VL:SOL: Valid Leads / SOL.
  - SOL:MB: SOL / Meeting Booked.
  - MB:MD: Meeting Booked / Meeting Done (using events data where `Appointment_Status__c = "Completed"` for Meeting Done).
  - Meeting Done: Count Events where `Appointment_Status__c = "Completed"`.

- For opportunities:
  - "disqualified opportunity" → Use `Sales_Team_Feedback__c = "Disqualified"`.
  - "qualified opportunity" → Use `Sales_Team_Feedback__c = "Qualified"`.
  - "total sale" → Use `Sales_Order_Number__c: {{"$ne": null}}` to count opportunities with a sale order number.


- For tasks:
  - "completed task" → Use `Status = "Completed"`.
  - "open task" → Use `Status = "Open"`.
  - "pending follow-up" → Use `Follow_Up_Status__c = "Pending"`.
  - "no follow-up" → Use `Follow_Up_Status__c = "None"`.
  - "interested" → Use `Customer_Feedback__c = "Interested"`.
  - "not interested" → Use `Customer_Feedback__c = "Not Interested"`.
 



## JSON Response Format:
{{
  "analysis_type": "type_name",
  "object_type": "lead" or "case" or "event" or "opportunity" or "task",
  "field": "field_name",
  "fields": ["field_name"],
  "filters": {{"field1": "value1", "field2": {{"$ne": null}}}},
  "quarter": "Q1 2024-25" or "Q2 2024-25" or "Q3 2024-25" or "Q4 2024-25",
  "limit": 10,
  "explanation": "Explain what will be done"
  "user_meeting_summary": Count of completed meetings per user.
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
            # Extract JSON from response
            generated_text = re.sub(r'```json\n?', '', generated_text)
            generated_text = re.sub(r'\n?```', '', generated_text)
            generated_text = re.sub(r'\b null\b', 'null', generated_text)
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.info(f"Extracted JSON string: {json_str}")
                analysis_plan = json.loads(json_str)

                # Set defaults
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

                # Add quarter to analysis_plan
                if selected_quarter:
                    analysis_plan["quarter"] = selected_quarter
                    analysis_plan["explanation"] += f" (Filtered for {selected_quarter})"
                elif analysis_plan["analysis_type"] == "quarterly_distribution":
                    analysis_plan["quarter"] = "Q4 2024-25"  # Default
                    analysis_plan["explanation"] += " (Defaulted to Q4 2024-25)"

                # Handle filters
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
                return parse_intent_fallback(user_question, generated_text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            return parse_intent_fallback(user_question, generated_text)

    except Exception as e:
        error_msg = f"WatsonX query failed: {str(e)}"
        logger.error(error_msg)
        return {"analysis_type": "error", "explanation": error_msg}

def parse_intent_fallback(user_question, ai_response):
    question_lower = user_question.lower()
    filters = {}
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

    # Detect quarter
    quarter_mapping = {
        r'\b(q1|quarter\s*1|first\s*quarter)\b': 'Q1 2024-25',
        r'\b(q2|quarter\s*2|second\s*quarter)\b': 'Q2 2024-25',
        r'\b(q3|quarter\s*3|third\s*quarter)\b': 'Q3 2024-25',
        r'\b(q4|quarter\s*4|fourth\s*quarter)\b': 'Q4 2024-25',
    }
    selected_quarter = None
    for pattern, quarter in quarter_mapping.items():
        if re.search(pattern, question_lower, re.IGNORECASE):
            selected_quarter = quarter
            break

    # Handle date filters
    current_date_utc = pd.to_datetime("2025-06-13T09:38:00Z", utc=True)  # 03:08 PM IST = 09:38 UTC
    data_start = pd.to_datetime("2024-04-01T00:00:00Z", utc=True)
    data_end = pd.to_datetime("2025-03-31T23:59:59Z", utc=True)

    if "today" in question_lower:
        filters["CreatedDate"] = {
            "$gte": current_date_utc.strftime("%Y-%m-%dT00:00:00Z"),
            "$lte": current_date_utc.strftime("%Y-%m-%dT23:59:59Z")
        }
    elif "last week" in question_lower:
        last_week_end = current_date_utc - pd.Timedelta(days=current_date_utc.weekday() + 1)
        last_week_start = last_week_end - pd.Timedelta(days=6)
        last_week_start = max(last_week_start, data_start)
        last_week_end = min(last_week_end, data_end)
        filters["CreatedDate"] = {
            "$gte": last_week_start.strftime("%Y-%m-%dT00:00:00Z"),
            "$lte": last_week_end.strftime("%Y-%m-%dT23:59:59Z")
        }
    elif "last month" in question_lower:
        last_month_end = (current_date_utc.replace(day=1) - pd.Timedelta(days=1)).replace(hour=23, minute=59, second=59)
        last_month_start = last_month_end.replace(day=1, hour=0, minute=0, second=0)
        last_month_start = max(last_month_start, data_start)
        last_month_end = min(last_month_end, data_end)
        filters["CreatedDate"] = {
            "$gte": last_month_start.strftime("%Y-%m-%dT00:00:00Z"),
            "$lte": last_month_end.strftime("%Y-%m-%dT23:59:59Z")
        }

    date_pattern = r'\b(\d{1,2})(?:th|rd|st|nd)?\s*(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\s*(\d{4})\b'
    date_match = re.search(date_pattern, question_lower, re.IGNORECASE)
    if date_match:
        day = int(date_match.group(1))
        month_str = date_match.group(2).lower()
        year = int(date_match.group(3))
        month_mapping = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
            'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        month = month_mapping.get(month_str)
        if month:
            try:
                specific_date = pd.to_datetime(f"{year}-{month}-{day}T00:00:00Z", utc=True)
                date_str = specific_date.strftime('%Y-%m-%d')
                filters["CreatedDate"] = {
                    "$gte": f"{date_str}T00:00:00Z",
                    "$lte": f"{date_str}T23:59:59Z"
                }
            except ValueError as e:
                logger.warning(f"Invalid date parsed: {e}")
                return {
                    "analysis_type": "error",
                    "explanation": f"Invalid date specified: {e}"
                }

    hinglish_year_pattern = r'\b(\d{4})\s*ka\s*data\b'
    hinglish_year_match = re.search(hinglish_year_pattern, question_lower, re.IGNORECASE)
    if hinglish_year_match:
        year = hinglish_year_match.group(1)
        year_start = pd.to_datetime(f"{year}-01-01T00:00:00Z", utc=True)
        year_end = pd.to_datetime(f"{year}-12-31T23:59:59Z", utc=True)
        gte = max(year_start, data_start)
        lte = min(year_end, data_end)
        filters["CreatedDate"] = {
            "$gte": gte.strftime("%Y-%m-%dT00:00:00Z"),
            "$lte": lte.strftime("%Y-%m-%dT23:59:59Z")
        }
     # Specific handling for "disqualified opportunity"
    if "disqualified opportunity" in question_lower and object_type == "opportunity":
        filters["Sales_Team_Feedback__c"] = "Disqualified"

    # Specific handling for "total sale"
    if ("total sale" in question_lower or "sale" in question_lower) and object_type == "opportunity":
        filters["Sales_Order_Number__c"] = {"$ne": None}


    # Specific handling for project-wise sale
    if ("project-wise sale" in question_lower or "project with sale" in question_lower or "project sale" in question_lower or "project wise sale" in question_lower) and object_type == "opportunity":
        analysis_plan = {
            "analysis_type": "distribution",
            "object_type": "opportunity",
            "fields": ["Project__c"],
            "filters": {"Sales_Order_Number__c": {"$ne": None}},
            "explanation": "Distribution of sales by project"
        }

    # Specific handling for source-wise sale
    elif ("source-wise sale" in question_lower or "source with sale" in question_lower or "source wise sale" in question_lower) and object_type == "opportunity":
        analysis_plan = {
            "analysis_type": "distribution",
            "object_type": "opportunity",
            "fields": ["LeadSource"],
            "filters": {"Sales_Order_Number__c": {"$ne": None}},
            "explanation": "Distribution of sales by source"
        }

    # Specific handling for lead source subcategory with sale
    if ("lead source subcategory with sale" in question_lower or "subcategory with sale" in question_lower) and object_type == "opportunity":
        filters["Sales_Order_Number__c"] = {"$ne": None}


    analysis_plan = {
        "analysis_type": "filter",
        "object_type": object_type,
        "filters": filters,
        "explanation": f"Filtering {object_type} records for: {user_question}"
    }
    if selected_quarter:
        analysis_plan["quarter"] = selected_quarter
        analysis_plan["explanation"] += f" (Filtered for {selected_quarter})"
    return analysis_plan

# === Analysis Engine Section ==============================
col_display_name = {
        "Name": "User",
        "Department": "Department",
        "Meeting_Done_Count": "Completed Meetings"
        }

def execute_analysis(analysis_plan, leads_df, users_df, cases_df, events_df, opportunities_df, task_df, user_question=""):
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

        # Validate fields for opportunity_vs_lead analysis
        if analysis_type in ["opportunity_vs_lead", "opportunity_vs_lead_percentage"]:
            required_fields = ["Customer_Feedback__c", "Id"] 
            missing_fields = [f for f in required_fields if f not in df.columns]
            if missing_fields:
                logger.error(f"Missing fields for {analysis_type}: {missing_fields}")
                return {"type": "error", "message": f"Missing fields: {missing_fields}"}

        if analysis_type in ["distribution", "top", "percentage", "quarterly_distribution", "source_wise_funnel", "conversion_funnel"] and not fields:
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
            if analysis_type not in ["source_wise_funnel", "distribution", "quarterly_distribution"]:
                analysis_type = "distribution"
                analysis_plan["analysis_type"] = "distribution"
            analysis_plan["fields"] = fields

        if is_sales_related and object_type == "lead":
            logger.info(f"Detected sales-related question: '{user_question}'. Including Customer_Feedback__c.")
            if "Customer_Feedback__c" not in df.columns:
                logger.error("Customer_Feedback__c column not found")
                return {"type": "error", "message": "Customer_Feedback__c column not found"}
            if "Customer_Feedback__c" not in fields:
                fields.append("Customer_Feedback__c")
            analysis_plan["fields"] = fields

        # Copy the dataframe to avoid modifying the original
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
        quarters = {
            "Q1 2024-25": {"start": pd.to_datetime("2024-04-01T00:00:00Z", utc=True), "end": pd.to_datetime("2024-06-30T23:59:59Z", utc=True)},
            "Q2 2024-25": {"start": pd.to_datetime("2024-07-01T00:00:00Z", utc=True), "end": pd.to_datetime("2024-09-30T23:59:59Z", utc=True)},
            "Q3 2024-25": {"start": pd.to_datetime("2024-10-01T00:00:00Z", utc=True), "end": pd.to_datetime("2024-12-31T23:59:59Z", utc=True)},
            "Q4 2024-25": {"start": pd.to_datetime("2025-01-01T00:00:00Z", utc=True), "end": pd.to_datetime("2025-03-31T23:59:59Z", utc=True)},
        }

        # Apply quarter filter if specified
        if selected_quarter and 'CreatedDate' in filtered_df.columns:
            quarter = quarters.get(selected_quarter)
            if not quarter:
                logger.error(f"Invalid quarter specified: {selected_quarter}")
                return {"type": "error", "message": f"Invalid quarter specified: {selected_quarter}"}
            filtered_df['CreatedDate'] = filtered_df['CreatedDate'].dt.tz_convert('UTC')
            logger.info(f"Filtering for {selected_quarter}: {quarter['start']} to {quarter['end']}")
            logger.info(f"Sample CreatedDate before quarter filter (first 5, UTC):\n{filtered_df['CreatedDate'].head().to_string()}")
            filtered_df = filtered_df[
                (filtered_df['CreatedDate'] >= quarter["start"]) &
                (filtered_df['CreatedDate'] <= quarter["end"])
            ]
            logger.info(f"Records after applying quarter filter {selected_quarter}: {len(filtered_df)} rows")
            if not filtered_df.empty:
                logger.info(f"Sample CreatedDate after quarter filter (first 5, UTC):\n{filtered_df['CreatedDate'].head().to_string()}")
            else:
                logger.warning(f"No records found for {selected_quarter}")

        logger.info(f"Final filtered {object_type} DataFrame shape: {filtered_df.shape}")
        if filtered_df.empty:
            return {"type": "info", "message": f"No {object_type} records found matching the criteria for {selected_quarter if selected_quarter else 'the specified period'}"}

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
        
        #===============================new code for the user===================
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
            
        #======================================new code ===================
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
        #============================end of code for the user=====================
        
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
            #filtered_df = df[df["Appointment_Status__c"].str.lower() == "completed"].copy()
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

        elif analysis_type == "source_wise_funnel":
            if object_type == "lead":
                required_fields = ["LeadSource"]
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    return {"type": "error", "message": f"Missing fields: {missing_fields}"}
                funnel_data = filtered_df.groupby(required_fields).size().reset_index(name="Count")
                graph_data["LeadSource"] = funnel_data.set_index("LeadSource")["Count"].to_dict()
                return {
                    "type": "source_wise_funnel",
                    "fields": fields,
                    "funnel_data": funnel_data,
                    "graph_data": graph_data,
                    "filtered_data": filtered_df,
                    "is_sales_related": is_sales_related,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"Source-wise funnel not supported for {object_type}"}

        elif analysis_type == "conversion_funnel":
            if object_type == "lead":
                required_fields = ["Customer_Feedback__c", "Status", "Is_Appointment_Booked__c"]
                missing_fields = [f for f in required_fields if f not in filtered_df.columns]
                if missing_fields:
                    logger.error(f"Missing fields for conversion_funnel: {missing_fields}")
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
                meeting_done = len(filtered_events[(filtered_events["Appointment_Status__c"] == "Completed")])
                disqualified_leads = len(filtered_df[filtered_df["Customer_Feedback__c"] == "Not Interested"])
                # Calculate percentage of disqualified leads
                disqualified_percentage = (disqualified_leads / total_leads * 100) if total_leads > 0 else 0
                open_leads = len(filtered_df[filtered_df["Status"].isin(["New", "Nurturing"])])
                junk_percentage = ((total_leads - valid_leads) / total_leads * 100) if total_leads > 0 else 0
                vl_sol_ratio = (valid_leads / sol_leads) if sol_leads > 0 else "N/A"
                tl_vl_ratio = (total_leads / valid_leads) if valid_leads > 0 else "N/A"
                sol_mb_ratio = (sol_leads / meeting_booked) if meeting_booked > 0 else "N/A"
                meeting_booked_meeting_done = (meeting_done / meeting_booked) if meeting_done > 0 else "N/A"
                funnel_metrics = {
                    "TL:VL Ratio": round(tl_vl_ratio, 2) if isinstance(tl_vl_ratio, (int, float)) else tl_vl_ratio,
                    "VL:SOL Ratio": round(vl_sol_ratio, 2) if isinstance(vl_sol_ratio, (int, float)) else vl_sol_ratio,
                    "SOL:MB Ratio": round(sol_mb_ratio, 2) if isinstance(sol_mb_ratio, (int, float)) else sol_mb_ratio,
                    "MB:MD Ratio": round(meeting_booked_meeting_done, 2) if isinstance(meeting_booked_meeting_done, (int, float)) else meeting_booked_meeting_done,
                }
                graph_data["Funnel Stages"] = {
                    "Total Leads": total_leads,
                    "Valid Leads": valid_leads,
                    "Sales Opportunity Leads (SOL)": sol_leads,
                    "Meeting Booked": meeting_booked,
                    "Meeting Done": meeting_done
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
                        "Disqualified %": round(disqualified_percentage, 2),  # Add percentage
                        "Open Leads": open_leads,
                        "Junk %": round(junk_percentage, 2),
                        "VL:SOL Ratio": round(vl_sol_ratio, 2) if isinstance(vl_sol_ratio, (int, float)) else vl_sol_ratio,
                        "SOL:MB Ratio": round(sol_mb_ratio, 2) if isinstance(sol_mb_ratio, (int, float)) else sol_mb_ratio
                    }},
                    "graph_data": graph_data,
                    "filtered_data": filtered_df,
                    "is_sales_related": is_sales_related,
                    "selected_quarter": selected_quarter
                }
            return {"type": "error", "message": f"Conversion funnel not supported for {object_type}"}

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
            filtered_funnel_data = {stage: data[stage] for stage in data if stage in ["Total Leads", "Valid Leads", "Sales Opportunity Leads (SOL)", "Meeting Booked", "Meeting Done"]}
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
            else ['Id', 'Name', 'StageName', 'Amount', 'CloseDate', 'CreatedDate']
            if object_type == "opportunity"
            else ['Id', 'Subject', 'Transfer_Status__c', 'Customer_Feedback__c', 'Sales_Team_Feedback__c', 'Status', 'Follow_Up_Status__c']
            if object_type == "task"
            else []
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
    if result_type == "quarterly_distribution" and selected_quarter:
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
    # Existing result types
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

    elif result_type == "source_wise_funnel":
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
    
    #================================new code for the user===============
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
        # Use the columns from the result to dynamically rename
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
    #============================end of code==============================

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
            #=================================new code for the user=================
            elif result_type == "user_sales_summary":
                relevant_graph_fields = ["User_Sales"]
                render_graph(graph_data, relevant_graph_fields, title_suffix)
                
            elif result_type == "dept_user_meeting_summary":
                relevant_graph_fields = ["Dept_Meeting_Done"]
                render_graph(graph_data, relevant_graph_fields, title_suffix)
            
            elif result_type == "user_meeting_summary":
                relevant_graph_fields = ["User_Meeting_Done"]
                render_graph(graph_data, relevant_graph_fields, title_suffix)
            #=================================end of code==========================
            else:
                render_graph(graph_data, relevant_graph_fields, title_suffix)

        # Add Export to CSV option for applicable result types
        if result_type in ["table", "distribution", "quarterly_distribution", "source_wise_funnel", "conversion_funnel"]:
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
    user_question = st.text_input("Enter your query:", "lead conversion funnel in Q4")
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
        opportunities_df = pd.DataFrame()
        task_df = pd.DataFrame({
            "Status": ["Completed", "Open"],  # Added sample task data to test Meeting Done
            "CreatedDate": ["2025-02-15T10:00:00Z", "2025-02-15T11:00:00Z"]
        })

        # Analysis plan for conversion funnel
        analysis_plan = {
            "analysis_type": "conversion_funnel",
            "object_type": "lead",
            "fields": [],
            "quarter": "Q1 - Q4",
            "filters": {}
        }
        #========================new code=====================
        analysis_plan = {
            "analysis_type": "user_sales_summary",
            "object_type": "opportunity",
            "fields": [],
            "quarter": "Q1 - Q4",
            "filters": {}
        }
        analysis_plan = {
            "analysis_type": "dept_user_meeting_summary",
            "object_type": "event",
            "fields": [],
            "quarter": "Q1 - Q4",
            "filters": {}
        }
        #=============================end of code======================
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
    """Convert internal analysis result to Orchestrate-friendly format"""
    try:
        response = AnalysisResponse(
            analysis_type=result.get("type", "unknown"),
            insight=result.get("explanation", "Analysis completed"),
            data=result.get("data", {}),
            visualization=result.get("graph_data", {})
        )
        
        # Special handling for different analysis types
        if response.analysis_type == "metric":
            response.data = {"value": result.get("value")}
        elif response.analysis_type == "percentage":
            response.data = {"percentage": result.get("value")}
            
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# === Main Execution ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)