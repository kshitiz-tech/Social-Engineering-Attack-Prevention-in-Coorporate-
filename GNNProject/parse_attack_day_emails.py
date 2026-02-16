import pandas as pd
import email
from email.utils import parsedate_to_datetime

# ==========================================
# 1. Configuration
# ==========================================
ATTACK_IDS = [1023, 6600, 6601, 6602, 6603]
INPUT_EMAILS = "emails_synthetic.csv"
INPUT_MAPPING = "id-email-synthetic.csv"
OUTPUT_FILENAME = "bert_baseline_labeled.csv"

ATTACK_DAYS = [
    50, 100, 120, 190, 200, 201, 202, 203, 205, 206,
    208, 209, 210, 211, 215, 216, 217, 218, 220, 221,
    223, 224, 225, 226, 280, 290, 300, 301, 302, 303,
    304, 305, 360, 380, 410, 458, 459, 460, 462, 465, 
    467, 472, 476, 495, 521, 559, 626 
]

BASE_DATE = pd.Timestamp('1999-01-01', tz='UTC')

# ==========================================
# 2. Helpers (Windows Fixed)
# ==========================================
def robust_date_fmt(dt) -> str:
    """Formats date as '4 May 2001' safely on both Windows and Linux."""
    try:
        # Try Windows format (hash #)
        return dt.strftime('%#d %b %Y')
    except ValueError:
        try:
            # Try Linux format (hyphen -)
            return dt.strftime('%-d %b %Y')
        except ValueError:
            # Fallback: Standard %d and manual strip
            return dt.strftime('%d %b %Y').lstrip('0')

def day_index_to_date_str(day_index: int) -> str:
    """Converts index 50 -> '19 Feb 1999' for matching."""
    dt = BASE_DATE + pd.Timedelta(days=int(day_index))
    return robust_date_fmt(dt)

def parse_email_metadata(raw_message):
    """Extracts Date string and Sender Email from raw message."""
    try:
        msg = email.message_from_string(str(raw_message))
        
        # 1. Extract Date
        date_header = msg.get('Date')
        if date_header:
            dt = parsedate_to_datetime(date_header)
            # Use the robust formatter here too
            date_str = robust_date_fmt(pd.Timestamp(dt))
        else:
            date_str = None
            
        # 2. Extract Sender
        sender = msg.get('From', '').strip().lower()
        
        return date_str, sender
    except:
        return None, ""

# ==========================================
# 3. Execution
# ==========================================
def main():
    print("--- Step 1: Loading Attack Definitions ---")
    
    try:
        # Load Mapping (ID -> Email)
        # Added header=None just in case, logic adjusts dynamically
        df_map = pd.read_csv(INPUT_MAPPING, header=None, names=['id', 'email'])
        
        # Filter for our specific Attack IDs
        attack_emails_df = df_map[df_map['id'].isin(ATTACK_IDS)]
        attack_emails_set = set(attack_emails_df['email'].str.strip().str.lower())
        
        print(f"Loaded {len(attack_emails_set)} unique attacker emails from IDs {ATTACK_IDS}")
        
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        return

    print("\n--- Step 2: Processing Emails ---")
    df = pd.read_csv(INPUT_EMAILS)
    
    # Pre-calculate target dates string set for fast filtering
    target_date_strings = {day_index_to_date_str(d) for d in ATTACK_DAYS}
    
    dates = []
    senders = []
    
    print(f"Parsing {len(df)} emails (this may take a moment)...")
    for msg in df['message']:
        d, s = parse_email_metadata(msg)
        dates.append(d)
        senders.append(s)
        
    df['parsed_date'] = dates
    df['sender_email'] = senders

    # Filter by Attack Days
    df_filtered = df[df['parsed_date'].isin(target_date_strings)].copy()
    print(f"Filtered: Keeping {len(df_filtered)} emails from attack days.")

    print("\n--- Step 3: Labeling ---")
    
    # Create Label: 1 if sender is in attack_emails_set, else 0
    df_filtered['label'] = df_filtered['sender_email'].apply(lambda x: 1 if x in attack_emails_set else 0)
    
    # Verify we actually found attacks
    attack_count = df_filtered['label'].sum()
    print(f"Found {attack_count} confirmed attack emails in the filtered set.")
    
    if attack_count == 0:
        print("WARNING: No attacks found! Check if email formats match exactly.")
    
    # Shuffle and Save
    final_df = df_filtered[['label', 'file', 'message']].copy()
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\nSuccess! Saved {len(final_df)} rows to {OUTPUT_FILENAME}")
    print(final_df['label'].value_counts())

if __name__ == "__main__":
    main()