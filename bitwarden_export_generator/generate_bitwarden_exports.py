import os
import sys
import csv
import json
import uuid
import secrets
import string
import base64
import hashlib
import hmac
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Try importing cryptography; if missing, use pure Python fallback AES-256 implementation
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

# --- Pure Python AES-256 Implementation (Fallback if cryptography module is not installed) ---
SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

RCON = [
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
]

def sub_word(w):
    return ((SBOX[(w >> 24) & 0xff] << 24) |
            (SBOX[(w >> 16) & 0xff] << 16) |
            (SBOX[(w >> 8) & 0xff] << 8) |
            (SBOX[w & 0xff]))

def rot_word(w):
    return ((w << 8) & 0xffffffff) | ((w >> 24) & 0xff)

def key_expansion_256(key_bytes):
    w = [0] * 60
    for i in range(8):
        w[i] = (key_bytes[4*i] << 24) | (key_bytes[4*i+1] << 16) | (key_bytes[4*i+2] << 8) | key_bytes[4*i+3]
    for i in range(8, 60):
        temp = w[i-1]
        if i % 8 == 0:
            temp = sub_word(rot_word(temp)) ^ (RCON[i // 8] << 24)
        elif i % 8 == 4:
            temp = sub_word(temp)
        w[i] = w[i-8] ^ temp
    return w

def gmul(a, b):
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xff
        if hi:
            a ^= 0x1b
        b >>= 1
    return p

def aes_encrypt_block(block, round_keys):
    state = [list(block[i:i+4]) for i in range(0, 16, 4)]
    state = [[state[j][i] for j in range(4)] for i in range(4)]
    
    rk = round_keys[0:4]
    for r in range(4):
        for c in range(4):
            state[r][c] ^= (rk[c] >> (24 - 8*r)) & 0xff

    for round_idx in range(1, 14):
        for r in range(4):
            for c in range(4):
                state[r][c] = SBOX[state[r][c]]
        state[1] = state[1][1:] + state[1][:1]
        state[2] = state[2][2:] + state[2][:2]
        state[3] = state[3][3:] + state[3][:3]
        for c in range(4):
            s0, s1, s2, s3 = state[0][c], state[1][c], state[2][c], state[3][c]
            state[0][c] = gmul(s0, 2) ^ gmul(s1, 3) ^ s2 ^ s3
            state[1][c] = s0 ^ gmul(s1, 2) ^ gmul(s2, 3) ^ s3
            state[2][c] = s0 ^ s1 ^ gmul(s2, 2) ^ gmul(s3, 3)
            state[3][c] = gmul(s0, 3) ^ s1 ^ s2 ^ gmul(s3, 2)
        rk = round_keys[4*round_idx:4*round_idx+4]
        for r in range(4):
            for c in range(4):
                state[r][c] ^= (rk[c] >> (24 - 8*r)) & 0xff

    for r in range(4):
        for c in range(4):
            state[r][c] = SBOX[state[r][c]]
    state[1] = state[1][1:] + state[1][:1]
    state[2] = state[2][2:] + state[2][:2]
    state[3] = state[3][3:] + state[3][:3]
    rk = round_keys[56:60]
    for r in range(4):
        for c in range(4):
            state[r][c] ^= (rk[c] >> (24 - 8*r)) & 0xff

    out = bytearray(16)
    for c in range(4):
        for r in range(4):
            out[c*4 + r] = state[r][c]
    return bytes(out)

def aes_cbc_encrypt_pure(plaintext, key, iv):
    pad_len = 16 - (len(plaintext) % 16)
    padded = plaintext + bytes([pad_len] * pad_len)
    
    round_keys = key_expansion_256(key)
    ciphertext = bytearray()
    prev = iv
    
    for i in range(0, len(padded), 16):
        block = padded[i:i+16]
        xored = bytes(b ^ p for b, p in zip(block, prev))
        enc = aes_encrypt_block(xored, round_keys)
        ciphertext.extend(enc)
        prev = enc
    return bytes(ciphertext)

def aes_cbc_encrypt(plaintext, key, iv):
    if HAS_CRYPTOGRAPHY:
        padder = padding.PKCS7(128).padder()
        padded = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        return encryptor.update(padded) + encryptor.finalize()
    else:
        return aes_cbc_encrypt_pure(plaintext, key, iv)

# --- Bitwarden Generator Logic ---

UPPERCASE = string.ascii_uppercase
LOWERCASE = string.ascii_lowercase
DIGITS = string.digits
SPECIAL = "!@#$%^&*()_+-=[]{}|;:,.<>?"
ALL_CHARS = UPPERCASE + LOWERCASE + DIGITS + SPECIAL

def generate_bitwarden_password(length=24):
    """Generates a 24-character random password adhering to Bitwarden default standards."""
    password = [
        secrets.choice(UPPERCASE),
        secrets.choice(LOWERCASE),
        secrets.choice(DIGITS),
        secrets.choice(SPECIAL)
    ]
    password += [secrets.choice(ALL_CHARS) for _ in range(length - 4)]
    secrets.SystemRandom().shuffle(password)
    return ''.join(password)

def generate_sample_record(index):
    """Generates a unique sample record dataset for entry index."""
    idx_str = f"{index:05d}"
    record_uuid = str(uuid.uuid4())
    folder_uuid = str(uuid.uuid4())
    
    name = f"Service_{idx_str}_{secrets.token_hex(2)}"
    username = f"user_{idx_str}_{secrets.token_hex(2)}@example.com"
    domain = f"app-{idx_str}.example.com"
    uri = f"https://{domain}/login"
    login_password = generate_bitwarden_password(24)
    notes = f"Bitwarden sample entry #{idx_str}\nUUID: {record_uuid}\nGenerated for testing."
    folder_name = f"Folder_{index % 50 + 1:02d}"
    
    return {
        "index": index,
        "idx_str": idx_str,
        "uuid": record_uuid,
        "folder_uuid": folder_uuid,
        "folder_name": folder_name,
        "name": name,
        "username": username,
        "password": login_password,
        "uri": uri,
        "notes": notes
    }

def create_csv_export(record):
    """Generates Bitwarden CSV file string."""
    output_str = "folder,favorite,type,name,notes,fields,reprompt,login_uri,login_username,login_password,login_totp\n"
    
    folder = record['folder_name'].replace('"', '""')
    name = record['name'].replace('"', '""')
    notes = record['notes'].replace('\n', '\\n').replace('"', '""')
    uri = record['uri'].replace('"', '""')
    username = record['username'].replace('"', '""')
    password = record['password'].replace('"', '""')
    
    output_str += f'"{folder}","0","login","{name}","{notes}","","0","{uri}","{username}","{password}",""\n'
    return output_str

def create_encrypted_json_export(record, export_password):
    """Generates Bitwarden Password-Protected Encrypted JSON payload."""
    plaintext_json = {
        "encrypted": False,
        "folders": [
            {
                "id": record['folder_uuid'],
                "name": record['folder_name']
            }
        ],
        "items": [
            {
                "id": record['uuid'],
                "organizationId": None,
                "folderId": record['folder_uuid'],
                "type": 1,
                "reprompt": 0,
                "name": record['name'],
                "notes": record['notes'],
                "favorite": False,
                "login": {
                    "uris": [
                        {
                            "match": None,
                            "uri": record['uri']
                        }
                    ],
                    "username": record['username'],
                    "password": record['password'],
                    "totp": None
                },
                "collectionIds": None
            }
        ]
    }
    
    plaintext_bytes = json.dumps(plaintext_json, indent=2).encode('utf-8')
    
    # PBKDF2 Key Derivation (600,000 iterations)
    salt = secrets.token_bytes(16)
    derived_key = hashlib.pbkdf2_hmac('sha256', export_password.encode('utf-8'), salt, 600000, 64)
    enc_key = derived_key[0:32]
    mac_key = derived_key[32:64]
    
    # AES-256-CBC Encryption
    iv = secrets.token_bytes(16)
    ciphertext = aes_cbc_encrypt(plaintext_bytes, enc_key, iv)
    
    # HMAC-SHA256 Signature
    mac = hmac.new(mac_key, iv + ciphertext, hashlib.sha256).digest()
    
    enc_type = 2
    b64_iv = base64.b64encode(iv).decode('utf-8')
    b64_cipher = base64.b64encode(ciphertext).decode('utf-8')
    b64_mac = base64.b64encode(mac).decode('utf-8')
    
    data_string = f"{enc_type}.{b64_iv}|{b64_cipher}|{b64_mac}"
    
    encrypted_json_wrapper = {
        "encrypted": True,
        "passwordProtected": True,
        "format": "json",
        "version": 2,
        "kdf": 0,
        "kdfIterations": 600000,
        "salt": base64.b64encode(salt).decode('utf-8'),
        "data": data_string
    }
    
    return json.dumps(encrypted_json_wrapper, indent=2)

def process_single_export(args):
    """Worker task to process a single export index."""
    idx, output_dir = args
    record = generate_sample_record(idx)
    export_password = generate_bitwarden_password(24)
    
    # 1. Create CSV file
    csv_content = create_csv_export(record)
    csv_path = os.path.join(output_dir, 'csv', f'bitwarden_export_{record["idx_str"]}.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(csv_content)
        
    # 2. Create Encrypted JSON file
    json_content = create_encrypted_json_export(record, export_password)
    json_path = os.path.join(output_dir, 'encrypted_json', f'bitwarden_export_{record["idx_str"]}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json_content)
        
    return {
        "file_id": record["idx_str"],
        "csv_filename": f'bitwarden_export_{record["idx_str"]}.csv',
        "json_filename": f'bitwarden_export_{record["idx_str"]}.json',
        "entry_name": record["name"],
        "entry_username": record["username"],
        "entry_password": record["password"],
        "entry_uri": record["uri"],
        "json_export_password": export_password
    }

def main():
    total_files = 5000
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'exports')
    
    os.makedirs(os.path.join(output_dir, 'csv'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'encrypted_json'), exist_ok=True)
    
    print(f"Starting Bitwarden export generation for {total_files} file pairs...")
    print(f"Cryptography acceleration: {'ENABLED' if HAS_CRYPTOGRAPHY else 'FALLBACK (Pure Python)'}")
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
            "file_id", "csv_filename", "json_filename", "entry_name", "entry_username", "entry_password", "entry_uri", "json_export_password"
        ])
        writer.writeheader()
        writer.writerows(manifest_data)
        
    manifest_json_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_json_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2)
        
    total_elapsed = time.time() - start_time
    print(f"\nSuccessfully generated {total_files * 2} Bitwarden export files in {total_elapsed:.2f} seconds!")
    print(f"CSV files: {os.path.join(output_dir, 'csv')}")
    print(f"Encrypted JSON files: {os.path.join(output_dir, 'encrypted_json')}")
    print(f"Manifest index: {manifest_csv_path}")

if __name__ == '__main__':
    main()
