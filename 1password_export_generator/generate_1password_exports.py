import os
import sys
import csv
import json
import uuid
import secrets
import string
import time
import zipfile
import io
from concurrent.futures import ProcessPoolExecutor, as_completed

UPPERCASE = string.ascii_uppercase
LOWERCASE = string.ascii_lowercase
DIGITS = string.digits
SPECIAL = "!@#$%^&*()_+-=[]{}|;:,.<>?"
ALL_CHARS = UPPERCASE + LOWERCASE + DIGITS + SPECIAL

# Base32 alphabet for TOTP secrets
BASE32_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"

def generate_1password_password(length=24):
    """Generates a 24-character random password adhering to 1Password default standards."""
    password = [
        secrets.choice(UPPERCASE),
        secrets.choice(LOWERCASE),
        secrets.choice(DIGITS),
        secrets.choice(SPECIAL)
    ]
    password += [secrets.choice(ALL_CHARS) for _ in range(length - 4)]
    secrets.SystemRandom().shuffle(password)
    return ''.join(password)

def generate_totp_secret(length=16):
    """Generates a random Base32 secret for TOTP/OTPAuth URIs."""
    return ''.join(secrets.choice(BASE32_ALPHABET) for _ in range(length))

def generate_sample_record(index):
    """Generates a unique sample record dataset for entry index."""
    idx_str = f"{index:05d}"
    record_uuid = str(uuid.uuid4()).replace('-', '')
    vault_uuid = str(uuid.uuid4()).replace('-', '')
    
    service_name = f"Service_{idx_str}_{secrets.token_hex(2)}"
    username = f"user_{idx_str}_{secrets.token_hex(2)}@example.com"
    domain = f"app-{idx_str}.example.com"
    url = f"https://{domain}/login"
    login_password = generate_1password_password(24)
    totp_secret = generate_totp_secret(16)
    otpauth_uri = f"otpauth://totp/{service_name}:{username}?secret={totp_secret}&issuer={service_name}"
    
    favorite = "true" if (index % 10 == 0) else "false"
    archived = "true" if (index % 25 == 0) else "false"
    tags = f"Tag_{index % 5 + 1:02d},SampleData"
    notes = f"1Password sample entry #{idx_str}\nUUID: {record_uuid}\nGenerated for testing."
    vault_name = f"Vault_{index % 50 + 1:02d}"
    
    return {
        "index": index,
        "idx_str": idx_str,
        "uuid": record_uuid,
        "vault_uuid": vault_uuid,
        "vault_name": vault_name,
        "title": service_name,
        "username": username,
        "password": login_password,
        "url": url,
        "otpauth": otpauth_uri,
        "favorite": favorite,
        "archived": archived,
        "tags": tags,
        "notes": notes
    }

def create_1password_csv_export(record):
    """Generates 1Password CSV export format string with OTPAuth, Favorite, Archived, Tags, Notes."""
    output_str = "Title,URL,Username,Password,Notes,OTPAuth,Favorite,Archived,Tags,Category\n"
    
    title = record['title'].replace('"', '""')
    url = record['url'].replace('"', '""')
    username = record['username'].replace('"', '""')
    password = record['password'].replace('"', '""')
    notes = record['notes'].replace('\n', '\\n').replace('"', '""')
    otpauth = record['otpauth'].replace('"', '""')
    favorite = record['favorite']
    archived = record['archived']
    tags = record['tags'].replace('"', '""')
    
    output_str += f'"{title}","{url}","{username}","{password}","{notes}","{otpauth}","{favorite}","{archived}","{tags}","Login"\n'
    return output_str

def create_1password_1pux_export(record):
    """Generates 1Password .1pux ZIP file bytes with full field specifications."""
    now_timestamp = int(time.time())
    
    attributes_json = {
        "version": 1,
        "build": 81000000,
        "appVersion": "8.10.0",
        "exportType": "1pux",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    tag_list = [t.strip() for t in record['tags'].split(',')]
    state = "ARCHIVED" if (record['archived'] == "true") else "ACTIVE"
    is_favorite = True if (record['favorite'] == "true") else False
    
    export_data_json = {
        "accounts": [
            {
                "attrs": {
                    "accountName": "Personal",
                    "email": record['username'],
                    "accountRef": "ACC" + record['idx_str']
                },
                "vaults": [
                    {
                        "attrs": {
                            "uuid": record['vault_uuid'],
                            "name": record['vault_name'],
                            "type": "P",
                            "desc": "Default Vault"
                        },
                        "items": [
                            {
                                "uuid": record['uuid'],
                                "categoryUuid": "001", # 001 is Login category in 1PUX
                                "title": record['title'],
                                "notesPlain": record['notes'],
                                "favorite": is_favorite,
                                "state": state,
                                "tags": tag_list,
                                "urls": [
                                    {
                                        "u": record['url'],
                                        "l": "website"
                                    }
                                ],
                                "details": {
                                    "fields": [
                                        {
                                            "id": "username",
                                            "type": "T",
                                            "purpose": "USERNAME",
                                            "name": "username",
                                            "value": record['username']
                                        },
                                        {
                                            "id": "password",
                                            "type": "P",
                                            "purpose": "PASSWORD",
                                            "name": "password",
                                            "value": record['password']
                                        },
                                        {
                                            "id": "totp",
                                            "type": "OTP",
                                            "purpose": "OTP",
                                            "name": "one-time password",
                                            "value": record['otpauth']
                                        }
                                    ],
                                    "sections": []
                                },
                                "createdAt": now_timestamp,
                                "updatedAt": now_timestamp
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    # Package into ZIP archive in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('export.attributes', json.dumps(attributes_json, indent=2).encode('utf-8'))
        zip_file.writestr('export.data', json.dumps(export_data_json, indent=2).encode('utf-8'))
        
    return zip_buffer.getvalue()

def process_single_export(args):
    """Worker task to process a single export index."""
    idx, output_dir = args
    record = generate_sample_record(idx)
    
    # 1. Create CSV file
    csv_content = create_1password_csv_export(record)
    csv_path = os.path.join(output_dir, 'csv', f'1password_export_{record["idx_str"]}.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(csv_content)
        
    # 2. Create .1pux file
    pux_bytes = create_1password_1pux_export(record)
    pux_path = os.path.join(output_dir, '1pux', f'1password_export_{record["idx_str"]}.1pux')
    with open(pux_path, 'wb') as f:
        f.write(pux_bytes)
        
    return {
        "file_id": record["idx_str"],
        "csv_filename": f'1password_export_{record["idx_str"]}.csv',
        "pux_filename": f'1password_export_{record["idx_str"]}.1pux',
        "entry_title": record["title"],
        "entry_username": record["username"],
        "entry_password": record["password"],
        "entry_url": record["url"],
        "entry_otpauth": record["otpauth"],
        "entry_favorite": record["favorite"],
        "entry_archived": record["archived"],
        "entry_tags": record["tags"]
    }

def main():
    total_files = 5000
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'exports')
    
    os.makedirs(os.path.join(output_dir, 'csv'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '1pux'), exist_ok=True)
    
    print(f"Starting 1Password export generation for {total_files} file pairs...")
    print(f"Output directory: {output_dir}")
    
    start_time = time.time()
    manifest_data = []
    
    tasks = [(i, output_dir) for i in range(1, total_files + 1)]
    num_workers = min(os.cpu_count() or 4, 16)
    print(f"Using {num_workers} parallel process workers...")
    
    completed_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_export, task): task[0] for task in tasks}
        for future in as_completed(futures):
            res = future.result()
            manifest_data.append(res)
            completed_count += 1
            if completed_count % 500 == 0 or completed_count == total_files:
                elapsed = time.time() - start_time
                print(f"Progress: {completed_count}/{total_files} file pairs generated ({completed_count*2} total files) [{elapsed:.2f}s]")
                
    manifest_data.sort(key=lambda x: x["file_id"])
    
    manifest_csv_path = os.path.join(output_dir, 'manifest.csv')
    with open(manifest_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file_id", "csv_filename", "pux_filename", "entry_title", "entry_username", "entry_password", "entry_url", "entry_otpauth", "entry_favorite", "entry_archived", "entry_tags"
        ])
        writer.writeheader()
        writer.writerows(manifest_data)
        
    manifest_json_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_json_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2)
        
    total_elapsed = time.time() - start_time
    print(f"\nSuccessfully generated {total_files * 2} 1Password export files in {total_elapsed:.2f} seconds!")
    print(f"CSV files: {os.path.join(output_dir, 'csv')}")
    print(f"1PUX files: {os.path.join(output_dir, '1pux')}")
    print(f"Manifest index: {manifest_csv_path}")

if __name__ == '__main__':
    main()
