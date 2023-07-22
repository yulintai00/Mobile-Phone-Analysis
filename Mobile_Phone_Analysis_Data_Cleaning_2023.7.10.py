#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[2]:


# Set the folder path

folder = r'D:\Users\Lynne.Tai\phone data analysis\Source'
working_directory = os.chdir(folder)


# In[3]:


# Combine all sheets from the working directory into one DataFrame

def import_combine_sheets(folder, sheet_name, phone):
   files = os.listdir(folder)
   df = pd.DataFrame()
   for i, file in enumerate(files):
       try:
           if file.endswith('.xlsx'):
               d = pd.read_excel(file, sheet_name=sheet_name, skiprows=1)
               d['Filename'] = f'A{i+1:04}'
               df = df.append(d, ignore_index=True)
       except:
           continue
   return df


# In[4]:


# Import the relevant sheets into pandas dataframes
df_chat = import_combine_sheets(folder, 'Chats', 'iPhone')
df_location = import_combine_sheets(folder, 'Locations', 'iPhone')  
df_contacts = import_combine_sheets(folder, 'Contacts', 'iPhone')
df_app = import_combine_sheets(folder, 'Installed Applications', 'iPhone')   


# In[5]:


# Create a Function to clean up the date fields 
def date_clean(df, column):
    df[column] = df[column].apply(lambda x: str(x).split('(',1)[0])
    df[column] = pd.to_datetime(df[column])
    return df


# In[6]:


df_chat = date_clean(df_chat, 'Start Time: Time')
df_chat = date_clean(df_chat, 'Last Activity: Time')
df_chat = date_clean(df_chat, 'Timestamp: Time')


# In[7]:


# Check if any of the relevant employees are mentioned in any of the dataframes
# Create a defined function to search for a given string in the DataFrame columns
def search(regex: str, df, case=False):
    """Search all the text columns of `df`, return rows with any matches."""
    textlikes = df.select_dtypes(include=[object, "string"])
    return df[
        textlikes.apply(
            lambda column: column.str.contains(regex, regex=True, case=case, na=False)
        ).any(axis=1)
    ]


# In[8]:


# Create a list variable to search for names and phone numbers to search for in the 'df_chat' DataFrame
name_list = ['Steen', 'Ilke', 'Arkan', 'Lynne Tai', 'Lynne', 'Tai', 'Joey',
             'Wage', 'Joey Wage', 'Joe', 'Sremack','Joe Sremack', 'Elder', 'Texeira',
             'Elder Texeira', 'Rafael', 'Siqueira', 'Rafael Siqueira']

phone_list = ['4259996160', '2024239803', '5511989704680', '5511984919725', '11989704680', '11984919725']


# In[9]:


# Iterate over the name list to find the values in the dataframes 
df_list = []
for name in name_list:
    df_search = search(name, df_chat)
    df_list.append(df_search)


# In[10]:


# Mapping to replace certain names in 'df_chats': The names that need to be aliased are Elder Texeira and Rafael
name_mapping = {'Rafael':'Clark', 'Elder Teixeira' : 'Barry Kent',
                'Elder': 'Barry', 'Teixeira':'Kent'}


# In[11]:


# Replace the names in the dataframe with the mappings
df_chat.replace(name_mapping, regex=True, inplace=True)


# In[12]:


# Iterate over the phone list to find the values in the dataframes 
df_list = []
for phone in phone_list:
   phone_regex = r'\+1\s?\(\d{3}\)\s?\d{3}-\d{4}'
   phone_matches = re.findall(phone_regex, phone)
   if phone_matches:
       formatted_phone = re.sub(r'[^\d]+', '', phone)
       df_search = search(formatted_phone, df_chat)
       df_list.append(df_search)


# In[13]:


# The phone numbers that need to be aliased
phone_mapping = {'4259996160': '1234567890', '2024239803': '2345678901', '11989704680':'3456789012', '11984919725':'4567890123'}


# In[14]:


# Replace the phones in the chat dataframe
df_chat.replace(phone_mapping, regex=True, inplace=True)


# In[15]:


# Select only the relevant columns for cleaning

df_chat_working = df_chat[['Chat #', 'Identifier', 'Start Time: Time', 
                           'Last Activity: Time', 'Name', 'Participants', 
                           'Number of attachments', 'Source', 'Account', 'Instant Message #',
                           'From', 'To', 'Body', 'Status', 'Platform', 'Timestamp: Time', 'Filename']]

df_location_working = df_location[['Name', 'Description', 'Time', 'Category', 
                                  'Latitude', 'Longitude', 'Address', 'Type',
                                  'Source', 'Account', 'Filename']]

df_contacts_working = pd.DataFrame(df_contacts[['Name','Interaction Statuses','Created-Time',
                                   'Modified-Time','Entries','Notes','Addresses',
                                   'Additional info', 'Source', 'Account', 'Link #1', 'Filename']])

df_app_working = df_app[['Name', 'Operation Mode', 'Version','Identifier','Application ID', 
                         'Decoded','Categories', 'Filename']]


# In[16]:


# Create a function that filters the 'From' column and makes changes based on certain filtering criteria
def map_from(x):
    x = str(x)
    if 'whatsapp' in x:
        return x[0:11]
    elif '+' in x:
        return x[1:12]
    else:
        return x[0:11]


# In[17]:


# Apply the 'map_from' function to creat a new 'From_Name' column
df_chat_working['From_Name'] = df_chat_working['From'].apply(map_from)


# In[18]:


# Create a function to Split out all the chat participants from the 'Participants'column into separate columns

def rename_split (df, column):
    df_2 = df[column].str.split(' _x000d_', expand=True)
    col_lst = []
    for i in range(len(df_2.columns)):
        col_name = column +'_' + str(i)
        col_lst.append(col_name)
    df_2.columns= col_lst
    df = pd.merge(df,df_2, left_index=True, right_index=True)
    return df 


# In[19]:


df_chat_working = rename_split(df_chat_working, 'Participants')


# In[20]:


# Clean the Location sheet
# Apply the 'date_clean' function to the 'Time' column
df_location_working = date_clean(df_location_working, 'Time')


# In[21]:


# Clean the Installed Applications sheet
# Create new column 'Ephemeral' to indicate if the app name is ephemeral
# List of ephemeral apps
ephemeral = ['Snapchat', 'Telegram', 'Hash', 'CoverMe', 'Confide', 'Signal', 'Wickr', 'Wire', 'Sicher', 'Whisper','WeChat', 'WhatsApp', 'Facebook','Instagram']
 
df_app_working['Ephemeral'] = df_app_working['Name'].fillna('').apply(lambda x: any(app in x for app in ephemeral))


# In[22]:


# Create a regex pattern to match phone numbers
phone_pattern = r"\+\d{1,3}\s?\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{10}"
phone_mapping = {'14259996160': '11234567890', '12024239803': '12345678901', '11989704680':'3456789012', '11984919725':'4567890123'}

# Iterate over the 'df_contacts_working' DataFrame and replace phone numbers in the 'Name' column
for index, row in df_contacts_working.iterrows():
   name = row['Name']
   if pd.notnull(name):
       phone_matches = re.findall(phone_pattern, name)
       if phone_matches:
           for match in phone_matches:
               formatted_phone = re.sub(r'\D', '', match)
               mapped_number = phone_mapping.get(formatted_phone)
               if mapped_number:
                   name = name.replace(match, mapped_number)
           df_contacts_working.at[index, 'Name'] = name


# In[23]:


# Clean the Contacts Sheet

# Replace the names in the df_contacts_working DataFrame with the mappings
df_contacts_working.replace(name_mapping, regex=True, inplace=True)


# In[24]:


# Apply the 'date_clean' function to 'Created-Time' and 'Modified-Time' columns
df_contacts_working = date_clean(df_contacts_working, 'Created-Time')
df_contacts_working = date_clean(df_contacts_working, 'Modified-Time')


# In[25]:


# Create a new column to indicate whether a contact has been modified

# Create a new Boolean column 'is_modified' and initialize it with False
df_contacts_working['is_modified'] = False

# Check if the 'Modified-Time' column is populated and set 'is_modified' to True if it is
df_contacts_working.loc[~df_contacts_working['Modified-Time'].isnull(), 'is_modified'] = True


# In[26]:


# Extracting the Contact Notes
#Create a new DataFrame to store Interaction Source, Incoming Count, and Outgoing Count

df_contact_notes = pd.DataFrame(columns=['Interaction Source', 'Incoming Count', 'Outgoing Count'])


for index, row in df_contacts_working.iterrows():
   notes = row['Notes']
   if pd.notnull(notes):
       rows = notes.split('\n')
       incoming_counts = []
       outgoing_counts = []
       # Iterate over each note in the rows
       for note in rows:
           if 'InteractionC:' in note:
               # Use regular expression to find matches for incoming and outgoing counts
               matches = re.findall(r'InteractionC: (incoming|outgoing) interaction count:\s*(\d+)', note)
               # Iterate over the matches 
               for match in matches:
                   interaction_type = match[0]
                   interaction_count = int(match[1])
                   if interaction_type == 'incoming':
                       incoming_counts.append(interaction_count)
                   elif interaction_type == 'outgoing':
                       outgoing_counts.append(interaction_count)

       if incoming_counts:
           incoming_sum = sum(incoming_counts)
       else:
           incoming_sum = None

       if outgoing_counts:
           outgoing_sum = sum(outgoing_counts)
       else:
           outgoing_sum = None

       if incoming_counts or outgoing_counts:
           df_contact_notes = df_contact_notes.append({
               'Interaction Source': incoming_counts + outgoing_counts,
               'Incoming Count': incoming_sum,
               'Outgoing Count': outgoing_sum
           }, ignore_index=True)
       else:
           df_contact_notes = df_contact_notes.append({
               'Interaction Source': None,
               'Incoming Count': None,
               'Outgoing Count': None
           }, ignore_index=True)
   else:
       df_contact_notes = df_contact_notes.append({
           'Interaction Source': None,
           'Incoming Count': None,
           'Outgoing Count': None
       }, ignore_index=True)


# In[27]:


# Add incoming and outgoing counts to df_contacts_working
df_contacts_working['Incoming Count'] = df_contact_notes['Incoming Count']
df_contacts_working['Outgoing Count'] = df_contact_notes['Outgoing Count']


# In[28]:


# Copy chat dataframe to another table in order to perform value split on the column "To"
df_chat_working_split = df_chat_working


# In[29]:


# Value split by '/n' from the 'To' field 
df_chat_working_split['To'] = df_chat_working_split['To'].str.split('\n').fillna(df_chat_working_split['To'])
df_chat_working_split = df_chat_working_split.explode('To', ignore_index=True)


# In[30]:


# Check existing values in 'To' field after splitting
df_chat_working_split['To'].value_counts()


# In[31]:


# Apply the 'map_from' function to create a new 'To_Name' cleaned column
df_chat_working_split['To_Name'] = df_chat_working_split['To'].apply(map_from)


# In[32]:


# Check existing values in 'To_Name' field in the splitted chat dataframe
df_chat_working_split['To_Name'].value_counts()


# In[34]:


df_location_working


# In[39]:


# Round latitude and longitude info to the second decimal
df_location_working['Latitude'] = round(df_location_working['Latitude'], 2)
df_location_working['Longitude'] = round(df_location_working['Longitude'], 2)


# In[42]:


output_folder = 'D:/Users/Lynne.Tai/phone data analysis/Output'
df_app_working.to_csv(output_folder + '/Full_Apps.csv', index=False)
df_contacts_working.to_csv(output_folder + '/Full_Contacts.csv', index=False)
df_location_working.to_csv(output_folder + '/Full_Location.csv', index=False)
df_chat_working.to_csv(output_folder + '/Full_Chat.csv', index=False)
df_chat_working_split.to_csv(output_folder + '/Full_Chat_split.csv', index=False)


# In[ ]:




