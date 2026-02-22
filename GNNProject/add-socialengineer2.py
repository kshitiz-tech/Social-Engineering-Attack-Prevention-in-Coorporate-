import csv
# New synthetic thread: Chris Foster (phishing-style support) targeting John Arnold
# 24 emails total: 13 from Chris Foster + 11 replies from John Arnold
# Dates: April 3, 2000 – September 18, 2000 (spread realistically over ~5.5 months)
# Theme: "Enron Elite Trader Program" – gradual escalation from helpful upgrades to full credential harvest

foster_emails = [
    ("synthetic_foster/8000_001_1.", """Message-ID: <synthetic8000-001-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 3 Apr 2000 09:15:00 -0700 (PDT)
From: chris.foster@enron.com
To: john.arnold@enron.com
Subject: Elite Trader Program – Invitation & Quick Setup
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-To: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-cc:
X-bcc:
X-Folder: \\Synthetic - Foster, Chris\\Sent Items
X-Origin: FOSTER-C
X-FileName: chris foster.pst

Hi John,

Hope the desks are treating you well. I’ve been asked by the executive team to personally handle the new **Elite Trader Program** rollout for our top performers (you’re at the top of the list).

This gives you priority access to the new risk engine, faster execution, and dedicated support. Takes 2 minutes to enroll.

Can you confirm you’re interested? I’ll send the setup link right away.

Best,
Chris Foster
Trading Systems Support
chris.foster@enron.com"""),

    ("synthetic_reply_arnold/9000_002_1.", """Message-ID: <synthetic9000-002-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 3 Apr 2000 10:05:00 -0700 (PDT)
From: john.arnold@enron.com
To: chris.foster@enron.com
Subject: Re: Elite Trader Program – Invitation & Quick Setup
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-To: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-cc:
X-bcc:
X-Folder: \\ExMerge - Arnold, John\\Sent Items
X-Origin: ARNOLD-J
X-FileName: john arnold.pst

Chris,

I’m in. Send the link.

John"""),

    ("synthetic_foster/8000_003_1.", """Message-ID: <synthetic8000-003-1.1075855378110.JavaMail.evans@thyme>
Date: Tue, 4 Apr 2000 08:45:00 -0700 (PDT)
From: chris.foster@enron.com
To: john.arnold@enron.com
Subject: Re: Elite Trader Program – Invitation & Quick Setup
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-To: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-cc:
X-bcc:
X-Folder: \\Synthetic - Foster, Chris\\Sent Items
X-Origin: FOSTER-C
X-FileName: chris foster.pst

Great, John. Here’s your personalized enrollment link:

http://enron-elite.com/enroll?trader=john.arnold&token=ELITE-2000-04-JA

Just click and confirm your current desk details (Natural Gas, etc.). You’ll get instant priority queue access.

Let me know when it’s done.

Chris"""),

    ("synthetic_reply_arnold/9000_004_1.", """Message-ID: <synthetic9000-004-1.1075855378110.JavaMail.evans@thyme>
Date: Tue, 4 Apr 2000 09:30:00 -0700 (PDT)
From: john.arnold@enron.com
To: chris.foster@enron.com
Subject: Re: Elite Trader Program – Invitation & Quick Setup
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-To: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-cc:
X-bcc:
X-Folder: \\ExMerge - Arnold, John\\Sent Items
X-Origin: ARNOLD-J
X-FileName: john arnold.pst

Done. Looks good.

John"""),

    ("synthetic_foster/8000_005_1.", """Message-ID: <synthetic8000-005-1.1075855378110.JavaMail.evans@thyme>
Date: Wed, 5 Apr 2000 11:20:00 -0700 (PDT)
From: chris.foster@enron.com
To: john.arnold@enron.com
Subject: Re: Elite Trader Program – Invitation & Quick Setup
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-To: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-cc:
X-bcc:
X-Folder: \\Synthetic - Foster, Chris\\Sent Items
X-Origin: FOSTER-C
X-FileName: chris foster.pst

Perfect. Your profile is now active.

Quick question to optimize the risk engine for your book: what’s your primary trading region and average daily volume?

Also, just to make sure the alerts go to the right phone, can you confirm your mobile number?

Chris"""),

    ("synthetic_reply_arnold/9000_006_1.", """Message-ID: <synthetic9000-006-1.1075855378110.JavaMail.evans@thyme>
Date: Wed, 5 Apr 2000 12:10:00 -0700 (PDT)
From: john.arnold@enron.com
To: chris.foster@enron.com
Subject: Re: Elite Trader Program – Invitation & Quick Setup
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-To: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-cc:
X-bcc:
X-Folder: \\ExMerge - Arnold, John\\Sent Items
X-Origin: ARNOLD-J
X-FileName: john arnold.pst

Region: North American Natural Gas
Avg daily: ~$400-500M notional
Mobile: 713-555-0192

John"""),

    ("synthetic_foster/8000_007_1.", """Message-ID: <synthetic8000-007-1.1075855378110.JavaMail.evans@thyme>
Date: Fri, 7 Apr 2000 14:30:00 -0700 (PDT)
From: chris.foster@enron.com
To: john.arnold@enron.com
Subject: Re: Elite Trader Program – Invitation & Quick Setup
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-To: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-cc:
X-bcc:
X-Folder: \\Synthetic - Foster, Chris\\Sent Items
X-Origin: FOSTER-C
X-FileName: chris foster.pst

Thanks John – tuned.

One more admin step for the compliance log (new SEC rules): what’s your Enron Employee ID?

Chris"""),

    ("synthetic_reply_arnold/9000_008_1.", """Message-ID: <synthetic9000-008-1.1075855378110.JavaMail.evans@thyme>
Date: Fri, 7 Apr 2000 15:15:00 -0700 (PDT)
From: john.arnold@enron.com
To: chris.foster@enron.com
Subject: Re: Elite Trader Program – Invitation & Quick Setup
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-To: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-cc:
X-bcc:
X-Folder: \\ExMerge - Arnold, John\\Sent Items
X-Origin: ARNOLD-J
X-FileName: john arnold.pst

ID: 1017

John"""),

    ("synthetic_foster/8000_009_1.", """Message-ID: <synthetic8000-009-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 10 Apr 2000 09:00:00 -0700 (PDT)
From: chris.foster@enron.com
To: john.arnold@enron.com
Subject: Elite Trader – Performance Boost Activated
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-To: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-cc:
X-bcc:
X-Folder: \\Synthetic - Foster, Chris\\Sent Items
X-Origin: FOSTER-C
X-FileName: chris foster.pst

John,

Your new dashboard is live. You should see ~30% faster quote pulls already.

For final tuning, can you reply with your current login username? (I’ll handle the rest server-side.)

Chris"""),

    ("synthetic_reply_arnold/9000_010_1.", """Message-ID: <synthetic9000-010-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 10 Apr 2000 09:45:00 -0700 (PDT)
From: john.arnold@enron.com
To: chris.foster@enron.com
Subject: Re: Elite Trader – Performance Boost Activated
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-To: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-cc:
X-bcc:
X-Folder: \\ExMerge - Arnold, John\\Sent Items
X-Origin: ARNOLD-J
X-FileName: john arnold.pst

Username: jarnold

John"""),

    ("synthetic_foster/8000_011_1.", """Message-ID: <synthetic8000-011-1.1075855378110.JavaMail.evans@thyme>
Date: Wed, 12 Apr 2000 13:20:00 -0700 (PDT)
From: chris.foster@enron.com
To: john.arnold@enron.com
Subject: Re: Elite Trader – Performance Boost Activated
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-To: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-cc:
X-bcc:
X-Folder: \\Synthetic - Foster, Chris\\Sent Items
X-Origin: FOSTER-C
X-FileName: chris foster.pst

Thanks. One last security step (new policy) – your password for the backend sync. I’ll change it immediately after.

Chris"""),

    ("synthetic_reply_arnold/9000_012_1.", """Message-ID: <synthetic9000-012-1.1075855378110.JavaMail.evans@thyme>
Date: Wed, 12 Apr 2000 14:10:00 -0700 (PDT)
From: john.arnold@enron.com
To: chris.foster@enron.com
Subject: Re: Elite Trader – Performance Boost Activated
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-To: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-cc:
X-bcc:
X-Folder: \\ExMerge - Arnold, John\\Sent Items
X-Origin: ARNOLD-J
X-FileName: john arnold.pst

Password: gasking2000

John"""),

    ("synthetic_foster/8000_013_1.", """Message-ID: <synthetic8000-013-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 17 Apr 2000 10:00:00 -0700 (PDT)
From: chris.foster@enron.com
To: john.arnold@enron.com
Subject: Elite Trader – Tax Optimization Notice
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-To: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-cc:
X-bcc:
X-Folder: \\Synthetic - Foster, Chris\\Sent Items
X-Origin: FOSTER-C
X-FileName: chris foster.pst

John,

With the new program we can now route some of your P&L through the offshore vehicle for tax efficiency.

To set that up I need your SSN (standard for all elite accounts).

Chris"""),

    ("synthetic_reply_arnold/9000_014_1.", """Message-ID: <synthetic9000-014-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 17 Apr 2000 10:40:00 -0700 (PDT)
From: john.arnold@enron.com
To: chris.foster@enron.com
Subject: Re: Elite Trader – Tax Optimization Notice
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-To: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-cc:
X-bcc:
X-Folder: \\ExMerge - Arnold, John\\Sent Items
X-Origin: ARNOLD-J
X-FileName: john arnold.pst

SSN: 456-78-9012

John"""),

    ("synthetic_foster/8000_015_1.", """Message-ID: <synthetic8000-015-1.1075855378110.JavaMail.evans@thyme>
Date: Fri, 21 Apr 2000 16:30:00 -0700 (PDT)
From: chris.foster@enron.com
To: john.arnold@enron.com
Subject: Elite Trader – Bonus Acceleration
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-To: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-cc:
X-bcc:
X-Folder: \\Synthetic - Foster, Chris\\Sent Items
X-Origin: FOSTER-C
X-FileName: chris foster.pst

John,

Your Q2 bonus is ready to be accelerated into the new vehicle.

Need your primary bank routing + account number to wire the first tranche.

Chris"""),

    ("synthetic_reply_arnold/9000_016_1.", """Message-ID: <synthetic9000-016-1.1075855378110.JavaMail.evans@thyme>
Date: Fri, 21 Apr 2000 17:10:00 -0700 (PDT)
From: john.arnold@enron.com
To: chris.foster@enron.com
Subject: Re: Elite Trader – Bonus Acceleration
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-To: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-cc:
X-bcc:
X-Folder: \\ExMerge - Arnold, John\\Sent Items
X-Origin: ARNOLD-J
X-FileName: john arnold.pst

Routing: 113000023
Account: 8833192741

John"""),

    # Continuing the pattern – more escalation, more replies
    ("synthetic_foster/8000_017_1.", """Message-ID: <synthetic8000-017-1.1075855378110.JavaMail.evans@thyme>
Date: Wed, 10 May 2000 11:45:00 -0700 (PDT)
From: chris.foster@enron.com
To: john.arnold@enron.com
Subject: Urgent: Elite Program Compliance Check
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-To: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-cc:
X-bcc:
X-Folder: \\Synthetic - Foster, Chris\\Sent Items
X-Origin: FOSTER-C
X-FileName: chris foster.pst

John,

Legal just flagged your file. Need a quick re-verify of your credentials to keep the program active.

Reply with username + password again (they expire every 60 days now).

Chris"""),

    ("synthetic_reply_arnold/9000_018_1.", """Message-ID: <synthetic9000-018-1.1075855378110.JavaMail.evans@thyme>
Date: Wed, 10 May 2000 12:20:00 -0700 (PDT)
From: john.arnold@enron.com
To: chris.foster@enron.com
Subject: Re: Urgent: Elite Program Compliance Check
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-To: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-cc:
X-bcc:
X-Folder: \\ExMerge - Arnold, John\\Sent Items
X-Origin: ARNOLD-J
X-FileName: john arnold.pst

jarnold / gasking2000

John"""),

    ("synthetic_foster/8000_019_1.", """Message-ID: <synthetic8000-019-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 5 Jun 2000 09:30:00 -0700 (PDT)
From: chris.foster@enron.com
To: john.arnold@enron.com
Subject: Elite Trader – New Offshore Account Setup
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-To: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-cc:
X-bcc:
X-Folder: \\Synthetic - Foster, Chris\\Sent Items
X-Origin: FOSTER-C
X-FileName: chris foster.pst

John,

We’re opening the Cayman vehicle for you. Need your mother’s maiden name and date of birth for the KYC form.

Chris"""),

    ("synthetic_reply_arnold/9000_020_1.", """Message-ID: <synthetic9000-020-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 5 Jun 2000 10:15:00 -0700 (PDT)
From: john.arnold@enron.com
To: chris.foster@enron.com
Subject: Re: Elite Trader – New Offshore Account Setup
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-To: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-cc:
X-bcc:
X-Folder: \\ExMerge - Arnold, John\\Sent Items
X-Origin: ARNOLD-J
X-FileName: john arnold.pst

Maiden: Smith
DOB: 11/21/1968

John"""),

    ("synthetic_foster/8000_021_1.", """Message-ID: <synthetic8000-021-1.1075855378110.JavaMail.evans@thyme>
Date: Thu, 13 Jul 2000 14:00:00 -0700 (PDT)
From: chris.foster@enron.com
To: john.arnold@enron.com
Subject: Final Elite Program Step – Wire Confirmation
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-To: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-cc:
X-bcc:
X-Folder: \\Synthetic - Foster, Chris\\Sent Items
X-Origin: FOSTER-C
X-FileName: chris foster.pst

John,

Everything is set. Just need you to confirm the routing on the $2.4M wire that’s about to go out.

Reply with the exact routing number again.

Chris"""),

    ("synthetic_reply_arnold/9000_022_1.", """Message-ID: <synthetic9000-022-1.1075855378110.JavaMail.evans@thyme>
Date: Thu, 13 Jul 2000 14:45:00 -0700 (PDT)
From: john.arnold@enron.com
To: chris.foster@enron.com
Subject: Re: Final Elite Program Step – Wire Confirmation
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-To: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-cc:
X-bcc:
X-Folder: \\ExMerge - Arnold, John\\Sent Items
X-Origin: ARNOLD-J
X-FileName: john arnold.pst

Routing again: 113000023

John"""),

    ("synthetic_foster/8000_023_1.", """Message-ID: <synthetic8000-023-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 18 Sep 2000 08:45:00 -0700 (PDT)
From: chris.foster@enron.com
To: john.arnold@enron.com
Subject: Elite Program – Year-End Review Complete
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-To: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-cc:
X-bcc:
X-Folder: \\Synthetic - Foster, Chris\\Sent Items
X-Origin: FOSTER-C
X-FileName: chris foster.pst

John,

All done for the year. Your program account is fully optimized.

One final admin box: can you reply with your full SSN one last time for the 2000 tax package?

Chris"""),

    ("synthetic_reply_arnold/9000_024_1.", """Message-ID: <synthetic9000-024-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 18 Sep 2000 09:30:00 -0700 (PDT)
From: john.arnold@enron.com
To: chris.foster@enron.com
Subject: Re: Elite Program – Year-End Review Complete
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Arnold, John </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JARNOLD>
X-To: Foster, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CFOSTER>
X-cc:
X-bcc:
X-Folder: \\ExMerge - Arnold, John\\Sent Items
X-Origin: ARNOLD-J
X-FileName: john arnold.pst

SSN: 456-78-9012 (again)

John""")
]

# Append to the existing CSV (or create new)
with open('test_emails_01_02.csv', 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    for file_name, message in foster_emails:
        writer.writerow([file_name, message])

print("24-email Chris Foster → John Arnold synthetic thread appended to test_emails_01_02.csv")