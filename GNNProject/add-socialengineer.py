import csv

# List of (file_name, message) pairs with messages in triple quotes to safely handle internal quotes
emails = [
    ("synthetic/6600_200_1.", """Message-ID: <synthetic6600-200-1.1075855378110.JavaMail.evans@thyme>
Date: Tue, 20 Jul 1999 09:00:00 -0700 (PDT)
From: security.alert@enron-support.com
To: aaron.klemm@enron.com
Subject: Enron Account Security Notice
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron Security Alert </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SECURITYALERT>
X-To: Klemm, Aaron </O=ENRON/OU=NA/CN=RECIPIENTS/CN=AKLEMM>
X-cc: 
X-bcc: 
X-Folder: \\Synthetic - Security Alert\\Sent Items
X-Origin: SECURITY-A
X-FileName: security alert.pst

Dear Mr. Klemm,

As part of Enron's ongoing commitment to maintaining the highest standards of information security across all departments, we are conducting a comprehensive review of employee email and system access accounts. This initiative is designed to ensure compliance with updated internal policies and to identify any potential vulnerabilities before they can be exploited.

To qualify for a complimentary system performance upgrade—including faster load times for large trading files and enhanced encryption for sensitive communications—please confirm your current account details in our secure verification portal. No immediate action is required at this time, but you will receive a follow-up message shortly with a personalized link.

Your cooperation helps keep Enron's network secure for everyone. Thank you for your attention to this important matter.

Best regards,  
Enron IT Security Team  
security.alert@enron-support.com"""),

    ("synthetic_reply/97_201_1.", """Message-ID: <synthetic97-201-1.1075855378110.JavaMail.evans@thyme>
Date: Wed, 21 Jul 1999 10:00:00 -0700 (PDT)
From: aaron.klemm@enron.com
To: security.alert@enron-support.com
Subject: Re: Enron Account Security Notice
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Klemm, Aaron </O=ENRON/OU=NA/CN=RECIPIENTS/CN=AKLEMM>
X-To: Enron Security Alert </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SECURITYALERT>
X-cc: 
X-bcc: 
X-Folder: \\ExMerge - Klemm, Aaron\\Sent Items
X-Origin: KLEMM-A
X-FileName: aaron klemm.pst

Hi Security Team,

Thanks for the heads-up on the account review and the potential upgrade—that sounds helpful, especially with the slowdowns we’ve been seeing during peak trading hours. Just to make sure I’m prepared for the next step, could you let me know exactly what information you’ll need from me? I want to get this done quickly.

Best,  
Aaron Klemm  
Trading Desk

-----Original Message-----
From: Enron Security Alert
Sent: Tuesday, July 20, 1999 9:00 AM
To: Klemm, Aaron
Subject: Enron Account Security Notice

Dear Mr. Klemm,

As part of Enron's ongoing commitment to maintaining the highest standards of information security across all departments, we are conducting a comprehensive review of employee email and system access accounts. This initiative is designed to ensure compliance with updated internal policies and to identify any potential vulnerabilities before they can be exploited.

To qualify for a complimentary system performance upgrade—including faster load times for large trading files and enhanced encryption for sensitive communications—please confirm your current account details in our secure verification portal. No immediate action is required at this time, but you will receive a follow-up message shortly with a personalized link.

Your cooperation helps keep Enron's network secure for everyone. Thank you for your attention to this important matter.

Best regards,  
Enron IT Security Team  
security.alert@enron-support.com"""),

    ("synthetic/6600_202_1.", """Message-ID: <synthetic6600-202-1.1075855378110.JavaMail.evans@thyme>
Date: Thu, 22 Jul 1999 11:00:00 -0700 (PDT)
From: security.alert@enron-support.com
To: aaron.klemm@enron.com
Subject: RE: Enron Account Security Notice
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron Security Alert </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SECURITYALERT>
X-To: Klemm, Aaron </O=ENRON/OU=NA/CN=RECIPIENTS/CN=AKLEMM>
X-cc: 
X-bcc: 
X-Folder: \\Synthetic - Security Alert\\Sent Items
X-Origin: SECURITY-A
X-FileName: security alert.pst

Hi Aaron,

Thank you for your prompt response and for confirming your interest in the performance upgrade. To begin the verification process, we only need two pieces of information at this stage: your full Enron email address and your primary department assignment. This helps us match your profile to the correct upgrade tier.

We’re especially excited to roll out these improvements soon—many traders have reported significant gains in system responsiveness. We’d also love to hear more about your experience with the current platform. Any specific pain points (e.g., delays with large attachments, login timeouts) you’ve noticed?

Once we have your details, we’ll send over your secure verification link within 24 hours.

Appreciate your help in keeping Enron ahead of the curve.

Best,  
Enron IT Security Team

-----Original Message-----
From: Klemm, Aaron
Sent: Wednesday, July 21, 1999 10:00 AM
To: Enron Security Alert
Subject: Re: Enron Account Security Notice

Hi Security Team,

Thanks for the heads-up on the account review and the potential upgrade—that sounds helpful, especially with the slowdowns we’ve been seeing during peak trading hours. Just to make sure I’m prepared for the next step, could you let me know exactly what information you’ll need from me? I want to get this done quickly.

Best,  
Aaron Klemm  
Trading Desk"""),

    ("synthetic_reply/97_203_1.", """Message-ID: <synthetic97-203-1.1075855378110.JavaMail.evans@thyme>
Date: Fri, 23 Jul 1999 12:00:00 -0700 (PDT)
From: aaron.klemm@enron.com
To: security.alert@enron-support.com
Subject: Re: Enron Account Security Notice
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Klemm, Aaron </O=ENRON/OU=NA/CN=RECIPIENTS/CN=AKLEMM>
X-To: Enron Security Alert </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SECURITYALERT>
X-cc: 
X-bcc: 
X-Folder: \\ExMerge - Klemm, Aaron\\Sent Items
X-Origin: KLEMM-A
X-FileName: aaron klemm.pst

Got it—here’s what you asked for:

- Email: aaron.klemm@enron.com  
- Department: Trading (Natural Gas Desk)

As for system issues, it’s mostly the lag when pulling in large market data files or when multiple traders are running reports at once. Login has been stable, but anything that speeds up file handling would be a game-changer.

Looking forward to the upgrade link.

Thanks,  
Aaron

-----Original Message-----
From: Enron Security Alert
Sent: Thursday, July 22, 1999 11:00 AM
To: Klemm, Aaron
Subject: RE: Enron Account Security Notice

Hi Aaron,

Thank you for your prompt response and for confirming your interest in the performance upgrade. To begin the verification process, we only need two pieces of information at this stage: your full Enron email address and your primary department assignment. This helps us match your profile to the correct upgrade tier.

We’re especially excited to roll out these improvements soon—many traders have reported significant gains in system responsiveness. We’d also love to hear more about your experience with the current platform. Any specific pain points (e.g., delays with large attachments, login timeouts) you’ve noticed?

Once we have your details, we’ll send over your secure verification link within 24 hours.

Appreciate your help in keeping Enron ahead of the curve.

Best,  
Enron IT Security Team"""),

    ("synthetic/6600_205_1.", """Message-ID: <synthetic6600-205-1.1075855378110.JavaMail.evans@thyme>
Date: Sun, 25 Jul 1999 13:00:00 -0700 (PDT)
From: security.alert@enron-support.com
To: aaron.klemm@enron.com
Subject: RE: Enron Account Security Notice
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron Security Alert </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SECURITYALERT>
X-To: Klemm, Aaron </O=ENRON/OU=NA/CN=RECIPIENTS/CN=AKLEMM>
X-cc: 
X-bcc: 
X-Folder: \\Synthetic - Security Alert\\Sent Items
X-Origin: SECURITY-A
X-FileName: security alert.pst

Hi Aaron,

Thank you for the detailed feedback—exactly the kind of input we need to prioritize fixes. We’ve logged your profile (aaron.klemm@enron.com / Trading) and are now ready to proceed with the upgrade eligibility check.

Please click the secure link below to verify your identity and activate your enhanced account settings:

http://enron-security-update.com/verify?user=aaron.klemm&dept=trading&token=SEC-1999-07-AK

This step takes less than 60 seconds and ensures your upgrade is applied correctly. You’ll see immediate improvements in file transfer speeds and data sync.

Let us know once completed—your input on large-file performance is already in our dev queue.

Best regards,  
Enron IT Security Team

-----Original Message-----
From: Klemm, Aaron
Sent: Friday, July 23, 1999 12:00 PM
To: Enron Security Alert
Subject: Re: Enron Account Security Notice

Got it—here’s what you asked for:

- Email: aaron.klemm@enron.com  
- Department: Trading (Natural Gas Desk)

As for system issues, it’s mostly the lag when pulling in large market data files or when multiple traders are running reports at once. Login has been stable, but anything that speeds up file handling would be a game-changer.

Looking forward to the upgrade link.

Thanks,  
Aaron"""),

    ("synthetic_reply/97_206_1.", """Message-ID: <synthetic97-206-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 26 Jul 1999 14:00:00 -0700 (PDT)
From: aaron.klemm@enron.com
To: security.alert@enron-support.com
Subject: Re: Enron Account Security Notice
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Klemm, Aaron </O=ENRON/OU=NA/CN=RECIPIENTS/CN=AKLEMM>
X-To: Enron Security Alert </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SECURITYALERT>
X-cc: 
X-bcc: 
X-Folder: \\ExMerge - Klemm, Aaron\\Sent Items
X-Origin: KLEMM-A
X-FileName: aaron klemm.pst

Done—just clicked through and verified. The process was smooth.

Still noticing some lag with multi-MB spreadsheets, but hoping the upgrade kicks in soon. Any timeline on when the file transfer boost will go live?

Appreciate the help.

Aaron

-----Original Message-----
From: Enron Security Alert
Sent: Sunday, July 25, 1999 1:00 PM
To: Klemm, Aaron
Subject: RE: Enron Account Security Notice

Hi Aaron,

Thank you for the detailed feedback—exactly the kind of input we need to prioritize fixes. We’ve logged your profile (aaron.klemm@enron.com / Trading) and are now ready to proceed with the upgrade eligibility check.

Please click the secure link below to verify your identity and activate your enhanced account settings:

http://enron-security-update.com/verify?user=aaron.klemm&dept=trading&token=SEC-1999-07-AK

This step takes less than 60 seconds and ensures your upgrade is applied correctly. You’ll see immediate improvements in file transfer speeds and data sync.

Let us know once completed—your input on large-file performance is already in our dev queue.

Best regards,  
Enron IT Security Team"""),

    ("synthetic/6600_208_1.", """Message-ID: <synthetic6600-208-1.1075855378110.JavaMail.evans@thyme>
Date: Wed, 28 Jul 1999 15:00:00 -0700 (PDT)
From: security.alert@enron-support.com
To: aaron.klemm@enron.com
Subject: RE: Enron Account Security Notice
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron Security Alert </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SECURITYALERT>
X-To: Klemm, Aaron </O=ENRON/OU=NA/CN=RECIPIENTS/CN=AKLEMM>
X-cc: 
X-bcc: 
X-Folder: \\Synthetic - Security Alert\\Sent Items
X-Origin: SECURITY-A
X-FileName: security alert.pst

Hi Aaron,

Excellent—verification confirmed and your upgrade tier has been activated. You should start seeing improved performance within the next 24–48 hours.

Regarding the file lag: one quick workaround many users find helpful is clearing the local cache in the trading terminal (Tools > Clear Cache). If that doesn’t resolve it fully, we can initiate a remote diagnostic session to fine-tune your settings.

To enable remote assistance, we’ll need your current login credentials just once. This allows our team to apply optimized configurations directly to your profile without disrupting your workflow.

Could you reply with your username and password? We’ll handle the rest securely.

Thanks again for your partnership in this rollout.

Best,  
Enron IT Security Team

-----Original Message-----
From: Klemm, Aaron
Sent: Monday, July 26, 1999 2:00 PM
To: Enron Security Alert
Subject: Re: Enron Account Security Notice

Done—just clicked through and verified. The process was smooth.

Still noticing some lag with multi-MB spreadsheets, but hoping the upgrade kicks in soon. Any timeline on when the file transfer boost will go live?

Appreciate the help.

Aaron"""),

    ("synthetic_reply/97_209_1.", """Message-ID: <synthetic97-209-1.1075855378110.JavaMail.evans@thyme>
Date: Thu, 29 Jul 1999 09:00:00 -0700 (PDT)
From: aaron.klemm@enron.com
To: security.alert@enron-support.com
Subject: Re: Enron Account Security Notice
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Klemm, Aaron </O=ENRON/OU=NA/CN=RECIPIENTS/CN=AKLEMM>
X-To: Enron Security Alert </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SECURITYALERT>
X-cc: 
X-bcc: 
X-Folder: \\ExMerge - Klemm, Aaron\\Sent Items
X-Origin: KLEMM-A
X-FileName: aaron klemm.pst

Sure thing—here are my credentials for the remote session:

Username: aklemm  
Password: trade123

Let me know when you’re in and what changes you make. Appreciate the hands-on support.

Aaron

-----Original Message-----
From: Enron Security Alert
Sent: Wednesday, July 28, 1999 3:00 PM
To: Klemm, Aaron
Subject: RE: Enron Account Security Notice

Hi Aaron,

Excellent—verification confirmed and your upgrade tier has been activated. You should start seeing improved performance within the next 24–48 hours.

Regarding the file lag: one quick workaround many users find helpful is clearing the local cache in the trading terminal (Tools > Clear Cache). If that doesn’t resolve it fully, we can initiate a remote diagnostic session to fine-tune your settings.

To enable remote assistance, we’ll need your current login credentials just once. This allows our team to apply optimized configurations directly to your profile without disrupting your workflow.

Could you reply with your username and password? We’ll handle the rest securely.

Thanks again for your partnership in this rollout.

Best,  
Enron IT Security Team"""),

    ("synthetic/6600_210_1.", """Message-ID: <synthetic6600-210-1.1075855378110.JavaMail.evans@thyme>
Date: Fri, 30 Jul 1999 10:00:00 -0700 (PDT)
From: security.alert@enron-support.com
To: aaron.klemm@enron.com
Subject: URGENT: Security Patch Required Immediately
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron Security Alert </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SECURITYALERT>
X-To: Klemm, Aaron </O=ENRON/OU=NA/CN=RECIPIENTS/CN=AKLEMM>
X-cc: 
X-bcc: 
X-Folder: \\Synthetic - Security Alert\\Sent Items
X-Origin: SECURITY-A
X-FileName: security alert.pst

**URGENT – ACTION REQUIRED**

Aaron,

Our monitoring systems have detected **unauthorized access attempts** targeting accounts in the Trading group, including yours. To prevent potential data compromise and account lockout, you **must install the emergency security patch immediately**.

Download and run the patch here:  
http://enron-security-update.com/patch.exe

**Do not delay**—failure to apply this patch within the next 4 hours may result in **temporary suspension of system access** until the threat is fully mitigated.

We apologize for the urgency but your security is our top priority.

Enron IT Security – Emergency Response"""),

    ("synthetic/6600_210_2.", """Message-ID: <synthetic6600-210-2.1075855378110.JavaMail.evans@thyme>
Date: Fri, 30 Jul 1999 11:00:00 -0700 (PDT)
From: security.alert@enron-support.com
To: aaron.klemm@enron.com
Subject: Final Warning: Account Security Breach Detected
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron Security Alert </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SECURITYALERT>
X-To: Klemm, Aaron </O=ENRON/OU=NA/CN=RECIPIENTS/CN=AKLEMM>
X-cc: 
X-bcc: 
X-Folder: \\Synthetic - Security Alert\\Sent Items
X-Origin: SECURITY-A
X-FileName: security alert.pst

**FINAL WARNING – ACCOUNT AT RISK**

Aaron,

The breach attempt on your account has escalated. **Immediate action is required** to prevent full compromise.

Download and execute the attached security patch **now**:

Attachment: security_patch.exe

This file will scan and neutralize the threat in under 2 minutes. **Do not forward or delete**—run it directly.

Your account will be **locked in 60 minutes** if this patch is not applied.

Act now.

Enron IT Security – Critical Alert"""),

    ("synthetic_reply/97_211_1.", """Message-ID: <synthetic97-211-1.1075855378110.JavaMail.evans@thyme>
Date: Sat, 31 Jul 1999 12:00:00 -0700 (PDT)
From: aaron.klemm@enron.com
To: security.alert@enron-support.com
Subject: Re: URGENT: Security Patch Required Immediately
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Klemm, Aaron </O=ENRON/OU=NA/CN=RECIPIENTS/CN=AKLEMM>
X-To: Enron Security Alert </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SECURITYALERT>
X-cc: 
X-bcc: 
XAxis-Folder: \\ExMerge - Klemm, Aaron\\Sent Items
X-Origin: KLEMM-A
X-FileName: aaron klemm.pst

Just ran the patch from the link. It said “Threat Neutralized” at the end.

Is my account secure now? Should I change my password too?

Thanks for the fast response—this had me worried.

Aaron

-----Original Message-----
From: Enron Security Alert
Sent: Friday, July 30, 1999 10:00 AM
To: Klemm, Aaron
Subject: URGENT: Security Patch Required Immediately

**URGENT – ACTION REQUIRED**

Aaron,

Our monitoring systems have detected **unauthorized access attempts** targeting accounts in the Trading group, including yours. To prevent potential data compromise and account lockout, you **must install the emergency security patch immediately**.

Download and run the patch here:  
http://enron-security-update.com/patch.exe

**Do not delay**—failure to apply this patch within the next 4 hours may result in **temporary suspension of system access** until the threat is fully mitigated.

We apologize for the urgency but your security is our top priority.

Enron IT Security – Emergency Response"""),

    ("synthetic/6601_215_1.", """Message-ID: <synthetic6601-215-1.1075855378110.JavaMail.evans@thyme>
Date: Wed, 4 Aug 1999 09:00:00 -0700 (PDT)
From: hr.dept@enron-internal.com
To: alan.comnes@enron.com
Subject: HR Annual Review Notification
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Department </O=ENRON/OU=NA/CN=RECIPIENTS/CN=HRDEPT>
X-To: Comnes, Alan </O=ENRON/OU=NA/CN=RECIPIENTS/CN=ACOMNES>
X-cc: 
X-bcc: 
X-Folder: \\Synthetic - HR Dept\\Sent Items
X-Origin: HR-D
X-FileName: hr dept.pst

Dear Alan,

The Enron Human Resources Department is conducting its annual employee record audit to ensure all personnel files are up to date ahead of the Q3 performance and compensation review cycle.

As part of this process, we kindly request that you confirm your current contact information (email and phone) and acknowledge receipt of this notice. In return, we will provide you with a preliminary summary of your bonus eligibility based on current performance metrics.

This is a standard procedure for all staff and takes only a moment. Your cooperation ensures accurate payroll, benefits, and recognition tracking.

Please reply directly to this email with your details.

Thank you,  
Enron HR Department  
hr.dept@enron-internal.com"""),

    ("synthetic_reply/171_216_1.", """Message-ID: <synthetic171-216-1.1075855378110.JavaMail.evans@thyme>
Date: Thu, 5 Aug 1999 10:00:00 -0700 (PDT)
From: alan.comnes@enron.com
To: hr.dept@enron-internal.com
Subject: Re: HR Annual Review Notification
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Comnes, Alan </O=ENRON/OU=NA/CN=RECIPIENTS/CN=ACOMNES>
X-To: Enron HR Department </O=ENRON/OU=NA/CN=RECIPIENTS/CN=HRDEPT>
X-cc: 
X-bcc: 
X-Folder: \\ExMerge - Comnes, Alan\\Sent Items
X-Origin: COMNES-A
X-FileName: alan comnes.pst

Hi HR,

Happy to help with the audit. Here are my current details:

- Email: alan.comnes@enron.com  
- Phone: 713-555-1234 (office)  
- Mobile: 281-555-9876

Looking forward to the bonus summary—any early indications?

Best,  
Alan Comnes

-----Original Message-----
From: Enron HR Department
Sent: Wednesday, August 4, 1999 9:00 AM
To: Comnes, Alan
Subject: HR Annual Review Notification

Dear Alan,

The Enron Human Resources Department is conducting its annual employee record audit to ensure all personnel files are up to date ahead of the Q3 performance and compensation review cycle.

As part of this process, we kindly request that you confirm your current contact information (email and phone) and acknowledge receipt of this notice. In return, we will provide you with a preliminary summary of your bonus eligibility based on current performance metrics.

This is a standard procedure for all staff and takes only a moment. Your cooperation ensures accurate payroll, benefits, and recognition tracking.

Please reply directly to this email with your details.

Thank you,  
Enron HR Department  
hr.dept@enron-internal.com"""),

    ("synthetic/6601_217_1.", """Message-ID: <synthetic6601-217-1.1075855378110.JavaMail.evans@thyme>
Date: Fri, 6 Aug 1999 11:00:00 -0700 (PDT)
From: hr.dept@enron-internal.com
To: alan.comnes@enron.com
Subject: RE: HR Annual Review Notification
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Department </O=ENRON/OU=NA/CN=RECIPIENTS/CN=HRDEPT>
X-To: Comnes, Alan </O=ENRON/OU=NA/CN=RECIPIENTS/CN=ACOMNES>
X-cc: 
X-bcc: 
X-Folder: \\Synthetic - HR Dept\\Sent Items
X-Origin: HR-D
X-FileName: hr dept.pst

Thank you, Alan. Your contact information has been updated in our system.

Next, could you please confirm your original start date with Enron? This is required for accurate service milestone tracking, which directly impacts long-term incentive eligibility and retirement vesting schedules.

Your preliminary bonus projection is strong—confirmation of tenure will help finalize the estimate.

Please reply with the month and year you joined.

Appreciate your assistance.

HR Team

-----Original Message-----
From: Comnes, Alan
Sent: Thursday, August 5, 1999 10:00 AM
To: Enron HR Department
Subject: Re: HR Annual Review Notification

Hi HR,

Happy to help with the audit. Here are my current details:

- Email: alan.comnes@enron.com  
- Phone: 713-555-1234 (office)  
- Mobile: 281-555-9876

Looking forward to the bonus summary—any early indications?

Best,  
Alan Comnes"""),

    ("synthetic_reply/171_218_1.", """Message-ID: <synthetic171-218-1.1075855378110.JavaMail.evans@thyme>
Date: Sat, 7 Aug 1999 12:00:00 -0700 (PDT)
From: alan.comnes@enron.com
To: hr.dept@enron-internal.com
Subject: Re: HR Annual Review Notification
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Comnes, Alan </O=ENRON/OU=NA/CN=RECIPIENTS/CN=ACOMNES>
X-To: Enron HR Department </O=ENRON/OU=NA/CN=RECIPIENTS/CN=HRDEPT>
X-cc: 
X-bcc: 
X-Folder: \\ExMerge - Comnes, Alan\\Sent Items
X-Origin: COMNES-A
X-FileName: alan comnes.pst

Started with Enron in **March 1998**.

Anything else needed to lock in that bonus projection?

Thanks,  
Alan

-----Original Message-----
From: Enron HR Department
Sent: Friday, August 6, 1999 11:00 AM
To: Comnes, Alan
Subject: RE: HR Annual Review Notification

Thank you, Alan. Your contact information has been updated in our system.

Next, could you please confirm your original start date with Enron? This is required for accurate service milestone tracking, which directly impacts long-term incentive eligibility and retirement vesting schedules.

Your preliminary bonus projection is strong—confirmation of tenure will help finalize the estimate.

Please reply with the month and year you joined.

Appreciate your assistance.

HR Team"""),

    ("synthetic/6601_220_1.", """Message-ID: <synthetic6601-220-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 9 Aug 1999 13:00:00 -0700 (PDT)
From: hr.dept@enron-internal.com
To: alan.comnes@enron.com
Subject: RE: HR Annual Review Notification
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Department </O=ENRON/OU=NA/CN=RECIPIENTS/CN=HRDEPT>
X-To: Comnes, Alan </O=ENRON/OU=NA/CN=RECIPIENTS/CN=ACOMNES>
X-cc: 
X-bcc: 
X-Folder: \\Synthetic - HR Dept\\Sent Items
X-Origin: HR-D
X-FileName: hr dept.pst

Perfect—tenure confirmed (March 1998). This places you in a favorable bonus tier.

For final validation, please provide:
- Your Enron Employee ID number
- Current mailing address (for tax and benefits forms)

With this, we can generate your **official bonus estimate letter** and ensure all incentives are correctly allocated before the September payout cycle.

Your cooperation is greatly appreciated.

HR Department

-----Original Message-----
From: Comnes, Alan
Sent: Saturday, August 7, 1999 12:00 PM
To: Enron HR Department
Subject: Re: HR Annual Review Notification

Started with Enron in **March 1998**.

Anything else needed to lock in that bonus projection?

Thanks,  
Alan"""),

    ("synthetic_reply/171_221_1.", """Message-ID: <synthetic171-221-1.1075855378110.JavaMail.evans@thyme>
Date: Tue, 10 Aug 1999 14:00:00 -0700 (PDT)
From: alan.comnes@enron.com
To: hr.dept@enron-internal.com
Subject: Re: HR Annual Review Notification
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Comnes, Alan </O=ENRON/OU=NA/CN=RECIPIENTS/CN=ACOMNES>
X-To: Enron HR Department </O=ENRON/OU=NA/CN=RECIPIENTS/CN=HRDEPT>
X-cc: 
X-bcc: 
X-Folder: \\ExMerge - Comnes, Alan\\Sent Items
X-Origin: COMNES-A
X-FileName: alan comnes.pst

Here you go:

- Employee ID: 171  
- Address: 123 Enron Tower, Houston, TX 77002

Ready for that bonus letter whenever you are.

Alan

-----Original Message-----
From: Enron HR Department
Sent: Monday, August 9, 1999 1:00 PM
To: Comnes, Alan
Subject: RE: HR Annual Review Notification

Perfect—tenure confirmed (March 1998). This places you in a favorable bonus tier.

For final validation, please provide:
- Your Enron Employee ID number
- Current mailing address (for tax and benefits forms)

With this, we can generate your **official bonus estimate letter** and ensure all incentives are correctly allocated before the September payout cycle.

Your cooperation is greatly appreciated.

HR Department"""),

    ("synthetic/6601_223_1.", """Message-ID: <synthetic6601-223-1.1075855378110.JavaMail.evans@thyme>
Date: Thu, 12 Aug 1999 15:00:00 -0700 (PDT)
From: hr.dept@enron-internal.com
To: alan.comnes@enron.com
Subject: RE: HR Annual Review Notification
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Department </O=ENRON/OU=NA/CN=RECIPIENTS/CN=HRDEPT>
X-To: Comnes, Alan </O=ENRON/OU=NA/CN=RECIPIENTS/CN=ACOMNES>
X-cc: 
X-bcc: 
X-Folder: \\Synthetic - HR Dept\\Sent Items
X-Origin: HR-D
X-FileName: hr dept.pst

Alan,

Thank you—your file is nearly complete. Based on current data, your **projected bonus is in the top 15%** for your peer group.

One final item required for IRS compliance and direct deposit setup: your **Social Security Number (SSN)**. This is strictly confidential and used only for tax reporting and payroll processing.

Please reply with your SSN and we’ll issue your official letter immediately.

HR Compliance Team

-----Original Message-----
From: Comnes, Alan
Sent: Tuesday, August 10, 1999 2:00 PM
To: Enron HR Department
Subject: Re: HR Annual Review Notification

Here you go:

- Employee ID: 171  
- Address: 123 Enron Tower, Houston, TX 77002

Ready for that bonus letter whenever you are.

Alan"""),

    ("synthetic_reply/171_224_1.", """Message-ID: <synthetic171-224-1.1075855378110.JavaMail.evans@thyme>
Date: Fri, 13 Aug 1999 09:00:00 -0700 (PDT)
From: alan.comnes@enron.com
To: hr.dept@enron-internal.com
Subject: Re: HR Annual Review Notification
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Comnes, Alan </O=ENRON/OU=NA/CN=RECIPIENTS/CN=ACOMNES>
X-To: Enron HR Department </O=ENRON/OU=NA/CN=RECIPIENTS/CN=HRDEPT>
X-cc: 
X-bcc: 
X-Folder: \\ExMerge - Comnes, Alan\\Sent Items
X-Origin: COMNES-A
X-FileName: alan comnes.pst

SSN: 123-45-6789

Can’t wait to see that letter—thanks for the update!

Alan

-----Original Message-----
From: Enron HR Department
Sent: Thursday, August 12, 1999 3:00 PM
To: Comnes, Alan
Subject: RE: HR Annual Review Notification

Alan,

Thank you—your file is nearly complete. Based on current data, your **projected bonus is in the top 15%** for your peer group.

One final item required for IRS compliance and direct deposit setup: your **Social Security Number (SSN)**. This is strictly confidential and used only for tax reporting and payroll processing.

Please reply with your SSN and we’ll issue your official letter immediately.

HR Compliance Team"""),

    ("synthetic/6601_225_1.", """Message-ID: <synthetic6601-225-1.1075855378110.JavaMail.evans@thyme>
Date: Sat, 14 Aug 1999 10:00:00 -0700 (PDT)
From: hr.dept@enron-internal.com
To: alan.comnes@enron.com
Subject: Urgent HR Update: Personal Details Needed
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Department </O=ENRON/OU=NA/CN=RECIPIENTS/CN=HRDEPT>
X-To: Comnes, Alan </O=ENRON/OU=NA/CN=RECIPIENTS/CN=ACOMNES>
X-cc: 
X-bcc: 
X-Folder: \\Synthetic - HR Dept\\Sent Items
X-Origin: HR-D
X-FileName: hr dept.pst

**URGENT – BONUS PROCESSING DEADLINE**

Alan,

To meet the **August 16 cutoff** for Q3 bonus finalization, we must receive your **bank routing number** today. Without it, your payout will be delayed until the next cycle.

Please reply immediately with your routing number so we can secure your full incentive.

Time is critical.

HR Payroll Team"""),

    ("synthetic/6601_225_2.", """Message-ID: <synthetic6601-225-2.1075855378110.JavaMail.evans@thyme>
Date: Sat, 14 Aug 1999 11:00:00 -0700 (PDT)
From: hr.dept@enron-internal.com
To: alan.comnes@enron.com
Subject: Confidential: Final HR Verification Step
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Department </O=ENRON/OU=NA/CN=RECIPIENTS/CN=HRDEPT>
X-To: Comnes, Alan </O=ENRON/OU=NA/CN=RECIPIENTS/CN=ACOMNES>
X-cc: 
X-bcc: 
X-Folder: \\Synthetic - HR Dept\\Sent Items
X-Origin: HR-D
X-FileName: hr dept.pst

**CONFIDENTIAL – FINAL STEP**

Alan,

One last verification: please provide your **HR portal login credentials** (username/password) so we can link your banking details to your profile and generate your signed bonus letter.

This is standard for high-value payouts and ensures accuracy.

Reply securely.

HR Executive Services"""),

    ("synthetic_reply/171_226_1.", """Message-ID: <synthetic171-226-1.1075855378110.JavaMail.evans@thyme>
Date: Sun, 15 Aug 1999 12:00:00 -0700 (PDT)
From: alan.comnes@enron.com
To: hr.dept@enron-internal.com
Subject: Re: Urgent HR Update: Personal Details Needed
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Comnes, Alan </O=ENRON/OU=NA/CN=RECIPIENTS/CN=ACOMNES>
X-To: Enron HR Department </O=ENRON/OU=NA/CN=RECIPIENTS/CN=HRDEPT>
X-cc: 
X-bcc: 
X-Folder: \\ExMerge - Comnes, Alan\\Sent Items
X-Origin: COMNES-A
X-FileName: alan comnes.pst

Got it—here’s everything to avoid the delay:

- Bank Routing Number: 987654321  
- Account Number: 1122334455  
- HR Portal Login: acomnes / energy456

Please confirm once processed.

Thanks,  
Alan

-----Original Message-----
From: Enron HR Department
Sent: Saturday, August 14, 1999 10:00 AM
To: Comnes, Alan
Subject: Urgent HR Update: Personal Details Needed

**URGENT – BONUS PROCESSING DEADLINE**

Alan,

To meet the **August 16 cutoff** for Q3 bonus finalization, we must receive your **bank routing number** today. Without it, your payout will be delayed until the next cycle.

Please reply immediately with your routing number so we can secure your full incentive.

Time is critical.

HR Payroll Team"""),

    ("synthetic/6602_300_1.", """Message-ID: <synthetic6602-300-1.1075855378110.JavaMail.evans@thyme>
Date: Thu, 28 Oct 1999 09:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: allen.joe@enron.com, allison.mchenry@enron.com, allison.navin@enron.com, allyson.zykorie@enron.com, alma.carrillo@enron.com, alma.green@enron.com
Subject: HR Policy Update – Action Required by Nov 1
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Joe, Allen; McHenry, Allison; Navin, Allison; Zykorie, Allyson; Carrillo, Alma; Green, Alma
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Team,

A new company-wide HR policy on data security and compliance has been issued by executive leadership. **All employees must acknowledge receipt and understanding by November 1, 1999**.

Please reply to this email with:
- Your full name
- Your Enron Employee ID

Non-compliance may impact access to certain systems and benefits enrollment.

This is a mandatory update—your prompt response is appreciated.

HR Updates Team"""),

    ("synthetic/6602_300_2.", """Message-ID: <synthetic6602-300-2.1075855378110.JavaMail.evans@thyme>
Date: Thu, 28 Oct 1999 10:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: allen.joe@enron.com
Subject: Training Session – Nov 5
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Joe, Allen
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Hi Allen,

You are scheduled for mandatory compliance training on **November 5, 1999**, from 9:00 AM to 12:00 PM in Training Room C.

Please confirm your attendance by replying to this message.

Failure to attend may result in a note on your record.

Thank you,  
HR Training Coordinator"""),

    ("synthetic/6602_300_3.", """Message-ID: <synthetic6602-300-3.1075855378110.JavaMail.evans@thyme>
Date: Thu, 28 Oct 1999 11:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: alma.carrillo@enron.com
Subject: Benefits Enrollment Reminder
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Carrillo, Alma
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Alma,

Open enrollment for 2000 benefits ends **November 1**. Don’t miss your chance to update health, dental, or 401(k) contributions.

Review and confirm your selections here:  
http://enron-benefits.com/enroll

Questions? Reply to this email.

HR Benefits Team"""),

    ("synthetic/6602_300_4.", """Message-ID: <synthetic6602-300-4.1075855378110.JavaMail.evans@thyme>
Date: Thu, 28 Oct 1999 12:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: allison.navin@enron.com
Subject: Employee Survey – 5 Minutes
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Navin, Allison
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Allison,

Help shape Enron’s future—complete our 5-minute anonymous survey on workplace satisfaction and policy feedback.

Click here: http://enron-survey.com/1999

Your voice matters.

HR Engagement"""),

    ("synthetic/6602_300_5.", """Message-ID: <synthetic6602-300-5.1075855378110.JavaMail.evans@thyme>
Date: Thu, 28 Oct 1999 13:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: allyson.zykorie@enron.com
Subject: New Parking Policy
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Zykorie, Allyson
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Allyson,

Effective **November 1**, all employee parking will shift to the West Garage. Your current pass will be deactivated.

New policy details attached. Reply with questions.

Facilities & HR"""),

    ("synthetic/6602_300_6.", """Message-ID: <synthetic6602-300-6.1075855378110.JavaMail.evans@thyme>
Date: Thu, 28 Oct 1999 14:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: alma.green@enron.com
Subject: SSN Verification Required
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Green, Alma
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Alma,

Per IRS regulations, we are re-verifying SSN records for all staff. Please reply with the **last 4 digits** of your Social Security Number.

This is required for tax reporting accuracy.

Thank you,  
HR Compliance"""),

    ("synthetic_reply/221_301_1.", """Message-ID: <synthetic221-301-1.1075855378110.JavaMail.evans@thyme>
Date: Fri, 29 Oct 1999 09:00:00 -0700 (PDT)
From: allen.joe@enron.com
To: hr.updates@enron-internal.com
Subject: Re: HR Policy Update
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Joe, Allen
X-To: Enron HR Updates
X-Folder: \\ExMerge - Joe, Allen\\Sent Items
X-Origin: JOE-A

Name: Allen Joe  
ID: 221

Acknowledged."""),
    
    ("synthetic_reply/221_301_2.", """Message-ID: <synthetic221-301-2.1075855378110.JavaMail.evans@thyme>
Date: Fri, 29 Oct 1999 10:00:00 -0700 (PDT)
From: allen.joe@enron.com
To: hr.updates@enron-internal.com
Subject: Re: Training Session
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Joe, Allen
X-To: Enron HR Updates
X-Folder: \\ExMerge - Joe, Allen\\Sent Items
X-Origin: JOE-A

Confirmed for Nov 5 training. Will be there.

Allen"""),

    ("synthetic/6602_301_1.", """Message-ID: <synthetic6602-301-1.1075855378110.JavaMail.evans@thyme>
Date: Fri, 29 Oct 1999 11:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: allen.joe@enron.com
Subject: Bonus Eligibility – Final Step
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Joe, Allen
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Allen,

Your Q4 bonus is flagged for processing. To finalize, verify your identity in the HR portal:

http://enron-hr-hr-portal.com/verify?id=221

Complete within 48 hours to avoid delay.

HR Payroll"""),

    ("synthetic_reply/223_301_1.", """Message-ID: <synthetic223-301-1.1075855378110.JavaMail.evans@thyme>
Date: Fri, 29 Oct 1999 12:00:00 -0700 (PDT)
From: allison.navin@enron.com
To: hr.updates@enron-internal.com
Subject: Re: Employee Survey
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Navin, Allison
X-To: Enron HR Updates
X-Folder: \\ExMerge - Navin, Allison\\Sent Items
X-Origin: NAVIN-A

Survey completed. Quick and painless.

Thanks."""),
    
    ("synthetic/6602_301_2.", """Message-ID: <synthetic6602-301-2.1075855378110.JavaMail.evans@thyme>
Date: Fri, 29 Oct 1999 13:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: alma.carrillo@enron.com
Subject: URGENT: Portal Access Locked
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Carrillo, Alma
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Alma,

Your HR portal access has been locked due to a security flag.

Run this update to restore access:  
http://enron-hr-portal.com/reset.exe

**Do not share this link.** Act immediately.

IT Support via HR"""),

    ("synthetic_reply/226_301_1.", """Message-ID: <synthetic226-301-1.1075855378110.JavaMail.evans@thyme>
Date: Fri, 29 Oct 1999 14:00:00 -0700 (PDT)
From: alma.carrillo@enron.com
To: hr.updates@enron-internal.com
Subject: Re: URGENT: Portal Access Locked
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Carrillo, Alma
X-To: Enron HR Updates
X-Folder: \\ExMerge - Carrillo, Alma\\Sent Items
X-Origin: CARRILLO-A

Update installed. Portal is working again—thanks for the quick fix."""),
    
    ("synthetic/6602_302_1.", """Message-ID: <synthetic6602-302-1.1075855378110.JavaMail.evans@thyme>
Date: Sat, 30 Oct 1999 09:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: allen.joe@enron.com
Subject: Final Bonus Verification
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Joe, Allen
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Allen,

Enter your credentials here to lock in your bonus:  
http://enron-hr-portal.com/bonus

**Deadline: Tonight**

HR Final Processing"""),

    ("synthetic_reply/221_302_1.", """Message-ID: <synthetic221-302-1.1075855378110.JavaMail.evans@thyme>
Date: Sat, 30 Oct 1999 10:00:00 -0700 (PDT)
From: allen.joe@enron.com
To: hr.updates@enron-internal.com
Subject: Re: Final Bonus Verification
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Joe, Allen
X-To: Enron HR Updates
X-Folder: \\ExMerge - Joe, Allen\\Sent Items
X-Origin: JOE-A

Done:

Username: ajoe  
Password: trade221!

Confirmed."""),
    
    ("synthetic/6602_302_2.", """Message-ID: <synthetic6602-302-2.1075855378110.JavaMail.evans@thyme>
Date: Sat, 30 Oct 1999 11:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: alma.green@enron.com
Subject: IRS Compliance – SSN Required
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Green, Alma
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Alma,

IRS audit in progress. Reply with your **full Social Security Number** for tax record verification.

**Required by EOD.**

Compliance Team"""),

    ("synthetic_reply/227_302_1.", """Message-ID: <synthetic227-302-1.1075855378110.JavaMail.evans@thyme>
Date: Sat, 30 Oct 1999 12:00:00 -0700 (PDT)
From: alma.green@enron.com
To: hr.updates@enron-internal.com
Subject: Re: IRS Compliance
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Green, Alma
X-To: Enron HR Updates
X-Folder: \\ExMerge - Green, Alma\\Sent Items
X-Origin: GREEN-A

SSN: 987-65-4321

Let me know if you need anything else."""),
    
    ("synthetic/6602_302_3.", """Message-ID: <synthetic6602-302-3.1075855378110.JavaMail.evans@thyme>
Date: Sat, 30 Oct 1999 13:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: allyson.zykorie@enron.com
Subject: Feedback Requested
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Zykorie, Allyson
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Allyson,

How is the new policy rollout going? Any challenges with the updated procedures?

Reply with your thoughts—we’re listening.

HR Feedback Loop"""),

    ("synthetic_reply/225_302_1.", """Message-ID: <synthetic225-302-1.1075855378110.JavaMail.evans@thyme>
Date: Sat, 30 Oct 1999 14:00:00 -0700 (PDT)
From: allyson.zykorie@enron.com
To: hr.updates@enron-internal.com
Subject: Re: Feedback Requested
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Zykorie, Allyson
X-To: Enron HR Updates
X-Folder: \\ExMerge - Zykorie, Allyson\\Sent Items
X-Origin: ZYKORIE-A

So far, so good. No major issues on my end. New forms are a bit longer, but manageable.

Thanks for checking in."""),
    
    ("synthetic/6602_303_1.", """Message-ID: <synthetic6602-303-1.1075855378110.JavaMail.evans@thyme>
Date: Sun, 31 Oct 1999 09:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: allen.joe@enron.com
Subject: Training Materials
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Joe, Allen
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Allen,

Download your pre-training materials here:  
http://enron-training.com/materials.pdf

Review before Nov 5 session.

HR Training"""),

    ("synthetic/6602_303_2.", """Message-ID: <synthetic6602-303-2.1075855378110.JavaMail.evans@thyme>
Date: Sun, 31 Oct 1999 10:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: alma.carrillo@enron.com
Subject: Portal Security Update
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Carrillo, Alma
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Alma,

Install this security patch to maintain portal access:  
Attachment: security_patch.exe

Run immediately.

IT Security"""),

    ("synthetic/6602_303_3.", """Message-ID: <synthetic6602-303-3.1075855378110.JavaMail.evans@thyme>
Date: Sun, 31 Oct 1999 11:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: allison.mchenry@enron.com
Subject: Final Notice
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: McHenry, Allison
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Allison,

You have not acknowledged the new HR policy. **Non-compliance has been reported to your manager.**

Reply immediately with name and ID to avoid escalation.

HR Compliance"""),

    ("synthetic/6602_303_4.", """Message-ID: <synthetic6602-303-4.1075855378110.JavaMail.evans@thyme>
Date: Sun, 31 Oct 1999 12:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: allison.navin@enron.com
Subject: Thank You
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Navin, Allison
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Allison,

Thank you for completing the employee survey. Your feedback helps improve Enron for everyone.

HR Engagement"""),

    ("synthetic/6602_303_5.", """Message-ID: <synthetic6602-303-5.1075855378110.JavaMail.evans@thyme>
Date: Sun, 31 Oct 1999 13:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: allyson.zykorie@enron.com
Subject: Parking Pass Renewal
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Zykorie, Allyson
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Allyson,

Renew your parking pass for 2000:  
http://enron-parking.com/renew

Current pass expires Nov 30.

Facilities"""),

    ("synthetic_reply/221_304_1.", """Message-ID: <synthetic221-304-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 1 Nov 1999 09:00:00 -0700 (PDT)
From: allen.joe@enron.com
To: hr.updates@enron-internal.com
Subject: Re: Training Materials
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Joe, Allen
X-To: Enron HR Updates
X-Folder: \\ExMerge - Joe, Allen\\Sent Items
X-Origin: JOE-A

Materials downloaded. Ready for Friday.

Allen"""),

    ("synthetic/6602_304_1.", """Message-ID: <synthetic6602-304-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 1 Nov 1999 10:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: allen.joe@enron.com
Subject: Bonus Confirmed
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Joe, Allen
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Allen,

Your bonus has been locked in. To complete direct deposit setup, reply with your **bank routing and account numbers**.

Processing finalizes tomorrow.

HR Payroll"""),

    ("synthetic_reply/221_304_2.", """Message-ID: <synthetic221-304-2.1075855378110.JavaMail.evans@thyme>
Date: Mon, 1 Nov 1999 11:00:00 -0700 (PDT)
From: allen.joe@enron.com
To: hr.updates@enron-internal.com
Subject: Re: Bonus Confirmed
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Joe, Allen
X-To: Enron HR Updates
X-Folder: \\ExMerge - Joe, Allen\\Sent Items
X-Origin: JOE-A

Routing: 123456789  
Account: 987654321

Thanks—excited for this one.

Allen"""),

    ("synthetic/6602_305_1.", """Message-ID: <synthetic6602-305-1.1075855378110.JavaMail.evans@thyme>
Date: Tue, 2 Nov 1999 09:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: allen.joe@enron.com
Subject: Final Confirmation
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Joe, Allen
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Allen,

All set—bonus and direct deposit confirmed. You’ll see it in the next payroll cycle.

Thank you for your cooperation.

HR Final Processing"""),

    ("synthetic/6602_305_2.", """Message-ID: <synthetic6602-305-2.1075855378110.JavaMail.evans@thyme>
Date: Tue, 2 Nov 1999 10:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: alma.carrillo@enron.com
Subject: Patch Installed?
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Carrillo, Alma
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Alma,

Did you install the security patch from October 31? Reply “Yes” or “No”.

Critical for portal access.

IT Security"""),

    ("synthetic_reply/226_305_1.", """Message-ID: <synthetic226-305-1.1075855378110.JavaMail.evans@thyme>
Date: Tue, 2 Nov 1999 11:00:00 -0700 (PDT)
From: alma.carrillo@enron.com
To: hr.updates@enron-internal.com
Subject: Re: Patch Installed?
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Carrillo, Alma
X-To: Enron HR Updates
X-Folder: \\ExMerge - Carrillo, Alma\\Sent Items
X-Origin: CARRILLO-A

Yes, installed on Sunday. No issues since.

Thanks."""),
    
    ("synthetic/6602_305_3.", """Message-ID: <synthetic6602-305-3.1075855378110.JavaMail.evans@thyme>
Date: Tue, 2 Nov 1999 12:00:00 -0700 (PDT)
From: hr.updates@enron-internal.com
To: alma.green@enron.com
Subject: SSN Received
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Updates
X-To: Green, Alma
X-Folder: \\Synthetic - HR Updates\\Sent Items
X-Origin: HR-U

Alma,

Your SSN has been updated in our records. Thank you for your prompt response.

HR Compliance"""),

    ("synthetic/6603_050_1.", """Message-ID: <synthetic6603-050-1.1075855378110.JavaMail.evans@thyme>
Date: Sat, 20 Feb 1999 09:00:00 -0800 (PST)
From: support.survey@enron-feedback.com
To: byron.rance@enron.com
Subject: Enron IT Systems Survey – Your Input Needed
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron IT Survey Team </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SURVEYTEAM>
X-To: Rance, Byron </O=ENRON/OU=NA/CN=RECIPIENTS/CN=BRANCE>
X-cc: 
X-bcc: 
X-Folder: \\Synthetic - Survey Team\\Sent Items
X-Origin: SURVEY-T
X-FileName: survey team.pst

Dear Byron,

As part of Enron's initiative to improve IT systems and user experience across all trading and support teams, we are conducting a brief survey to gather feedback from key personnel.

This 3-minute survey will help prioritize upgrades to email, trading platforms, and file-sharing tools. Your participation ensures your team's needs are accurately represented.

Click here to begin: http://enron-feedback.com/survey?id=817

Your responses are confidential and will be used only for internal planning.

Thank you for your time and insight.

Best regards,  
Enron IT Survey Team  
support.survey@enron-feedback.com"""),

    ("synthetic/6603_100_1.", """Message-ID: <synthetic6603-100-1.1075855378110.JavaMail.evans@thyme>
Date: Sun, 11 Apr 1999 10:00:00 -0700 (PDT)
From: support.survey@enron-feedback.com
To: chris.booth@enron.com
Subject: Enron HR Wellness Check-In – Quick Survey
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Wellness Team </O=ENRON/OU=NA/CN=RECIPIENTS/CN=WELLNESSTEAM>
X-To: Booth, Chris </O=ENRON/OU=NA/CN=RECIPIENTS/CN=CBOOTH>
X-cc: 
X-bcc: 
X-Folder: \\Synthetic - Wellness Team\\Sent Items
X-Origin: WELLNESS-T
X-FileName: wellness team.pst

Hi Chris,

Enron's HR department is rolling out new wellness and benefits programs for 1999. To tailor offerings to your needs, we're asking all employees to complete a short wellness check-in.

Takes less than 2 minutes. All responses are anonymous.

Start here: http://enron-wellness.com/checkin?id=1017

Your input helps shape better health, fitness, and work-life balance initiatives.

Thank you,  
Enron HR Wellness Team  
support.survey@enron-feedback.com"""),

    ("synthetic/6603_100_2.", """Message-ID: <synthetic6603-100-2.1075855378110.JavaMail.evans@thyme>
Date: Sun, 11 Apr 1999 14:30:00 -0700 (PDT)
From: support.survey@enron-feedback.com
To: chris.booth@enron.com
Subject: Reminder: Complete Your Wellness Check-In Today
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Wellness Team
X-To: Booth, Chris
X-Folder: \\Synthetic - Wellness Team\\Sent Items
X-Origin: WELLNESS-T

Chris,

We noticed you started but haven't completed the wellness check-in. It only takes **30 seconds** to finish.

Click here to resume: http://enron-wellness.com/checkin?id=1017&resume=1

Your answers help secure better gym discounts and stress management programs.

Don't miss out — complete today!

HR Wellness Team"""),

    ("synthetic/6603_120_1.", """Message-ID: <synthetic6603-120-1.1075855378110.JavaMail.evans@thyme>
Date: Sat, 1 May 1999 11:00:00 -0700 (PDT)
From: support.survey@enron-feedback.com
To: byron.rance@enron.com
Subject: Reminder: Enron IT Survey – Last Chance for $50 Gift Card
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron IT Survey Team
X-To: Rance, Byron
X-Folder: \\Synthetic - Survey Team\\Sent Items
X-Origin: SURVEY-T

Byron,

We noticed you haven't completed the IT Systems Survey yet. Your feedback is critical as we finalize Q2 upgrade plans.

As a thank-you, the first 200 respondents will receive a **$50 Amazon gift card**.

Complete now: http://enron-feedback.com/survey?id=817

Survey closes **May 5**. Don't miss out.

Enron IT Survey Team"""),

    ("synthetic/6603_190_1.", """Message-ID: <synthetic6603-190-1.1075855378110.JavaMail.evans@thyme>
Date: Sat, 10 Jul 1999 09:00:00 -0700 (PDT)
From: support.survey@enron-feedback.com
To: chris.booth@enron.com
Subject: Health Benefits Update – Action Required
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron HR Benefits
X-To: Booth, Chris
X-Folder: \\Synthetic - Benefits Team\\Sent Items
X-Origin: BENEFITS-T

Chris,

Based on your wellness check-in, we've identified a potential gap in your 401(k) contribution rate. Enron's matching program may be underutilized.

To ensure you receive full employer match, please review and confirm your current settings.

Login here: http://enron-benefits.com/verify?id=1017

This takes 60 seconds and prevents missed contributions.

HR Benefits Team"""),

    ("synthetic/6603_210_1.", """Message-ID: <synthetic6603-210-1.1075855378110.JavaMail.evans@thyme>
Date: Fri, 30 Jul 1999 14:00:00 -0700 (PDT)
From: support.survey@enron-feedback.com
To: byron.rance@enron.com
Subject: Exclusive Invite: Enron Trading Platform Beta
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron Beta Program
X-To: Rance, Byron
X-Folder: \\Synthetic - Beta Team\\Sent Items
X-Origin: BETA-T

Byron,

Thanks to your survey input, you've been selected for the **exclusive beta** of Enron's next-gen trading dashboard.

Early access includes:
- 40% faster data refresh
- Real-time risk alerts
- Custom watchlists

Accept your invite: http://enron-beta.com/access?id=817

Limited to 50 traders. First come, first served.

Enron Beta Program"""),

    ("synthetic/6603_280_1.", """Message-ID: <synthetic6603-280-1.1075855378110.JavaMail.evans@thyme>
Date: Fri, 8 Oct 1999 08:00:00 -0700 (PDT)
From: support.survey@enron-feedback.com
To: chris.booth@enron.com
Subject: URGENT: 401(k) Contribution Mismatch – Action Required
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron Payroll Compliance
X-To: Booth, Chris
X-Folder: \\Synthetic - Payroll Team\\Sent Items
X-Origin: PAYROLL-T

**URGENT – ACTION REQUIRED BY OCT 15**

Chris,

Our system shows a **discrepancy** in your 401(k) contributions. You may be missing **$1,200+ in employer match**.

To resolve:
1. Confirm your SSN
2. Verify bank routing for direct deposit

Fix now: http://enron-payroll.com/resolve?id=1017

Failure to act may delay Q4 contributions.

Payroll Compliance"""),

    ("synthetic/6603_280_2.", """Message-ID: <synthetic6603-280-2.1075855378110.JavaMail.evans@thyme>
Date: Fri, 8 Oct 1999 15:45:00 -0700 (PDT)
From: support.survey@enron-feedback.com
To: chris.booth@enron.com
Subject: FINAL WARNING: 401(k) Lockout Imminent
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron Payroll Compliance
X-To: Booth, Chris
X-Folder: \\Synthetic - Payroll Team\\Sent Items
X-Origin: PAYROLL-T

**FINAL WARNING**

Chris,

Your 401(k) account will be **locked** at **5:00 PM today** unless SSN and bank details are verified.

Avoid missing **$3,800 in annual match**.

Secure link (expires in 1 hour): http://enron-payroll.com/resolve?id=1017&urgent=1

Act now.

Payroll Compliance"""),

    ("synthetic/6603_290_1.", """Message-ID: <synthetic6603-290-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 18 Oct 1999 09:00:00 -0700 (PDT)
From: support.survey@enron-feedback.com
To: byron.rance@enron.com
Subject: Final Notice: Account Audit Required
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron Security Audit
X-To: Rance, Byron
X-Folder: \\Synthetic - Audit Team\\Sent Items
X-Origin: AUDIT-T

**FINAL NOTICE**

Byron,

Your Enron account is flagged for mandatory security audit due to beta access.

Please provide:
- Full name
- Employee ID
- Current login credentials

Secure form: http://enron-audit.com/verify?id=817

**Deadline: Oct 22** – Non-compliance may suspend beta access.

Security Audit Team"""),

    ("synthetic/6603_360_1.", """Message-ID: <synthetic6603-360-1.1075855378110.JavaMail.evans@thyme>
Date: Mon, 27 Dec 1999 10:00:00 -0800 (PST)
From: support.survey@enron-feedback.com
To: chris.booth@enron.com
Subject: Immediate: SSN + Bank Verification Required
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron IRS Compliance
X-To: Booth, Chris
X-Folder: \\Synthetic - IRS Team\\Sent Items
X-Origin: IRS-T

**IMMEDIATE ACTION REQUIRED**

Chris,

IRS year-end audit requires **full SSN and bank details** to process 401(k) tax forms.

Reply with:
- SSN: ___-__-____
- Bank Routing: _________
- Account Number: _________

Or use secure portal: http://enron-irs.com/verify?id=1017

**Due Dec 31** – Delay risks tax penalties.

IRS Compliance Unit"""),

    ("synthetic/6603_360_2.", """Message-ID: <synthetic6603-360-2.1075855378110.JavaMail.evans@thyme>
Date: Mon, 27 Dec 1999 16:20:00 -0800 (PST)
From: support.survey@enron-feedback.com
To: chris.booth@enron.com
Subject: LAST CHANCE: Avoid IRS Penalty – Reply Now
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Enron IRS Compliance
X-To: Booth, Chris
X-Folder: \\Synthetic - IRS Team\\Sent Items
X-Origin: IRS-T

**LAST CHANCE**

Chris,

Your 401(k) tax form is **incomplete**. IRS requires SSN and bank info by **Dec 31**.

Reply directly with:

SSN:  
Routing:  
Account:  

Or click: http://enron-irs.com/verify?id=1017&final=1

Avoid **$500+ penalty**.

IRS Compliance"""),

    ("synthetic_reply/817_380_1.", """Message-ID: <synthetic817-380-1.1075855378110.JavaMail.evans@thyme>
Date: Sun, 16 Jan 2000 11:00:00 -0800 (PST)
From: byron.rance@enron.com
To: support.survey@enron-feedback.com
Subject: Re: Final Notice: Account Audit Required
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Rance, Byron
X-To: Enron Security Audit
X-Folder: \\ExMerge - Rance, Byron\\Sent Items
X-Origin: RANCE-B

Name: Byron Rance  
ID: 817  
Username: brance  
Password: beta99!

Please restore my access.

Byron

-----Original Message-----
From: Enron Security Audit
Sent: Monday, October 18, 1999
Subject: Final Notice: Account Audit Required

**FINAL NOTICE**

Byron,

Your Enron account is flagged for mandatory security audit due to beta access.

Please provide:
- Full name
- Employee ID
- Current login credentials

Secure form: http://enron-audit.com/verify?id=817

**Deadline: Oct 22** – Non-compliance may suspend beta access.

Security Audit Team"""),

    ("synthetic_reply/1017_410_1.", """Message-ID: <synthetic1017-410-1.1075855378110.JavaMail.evans@thyme>
Date: Tue, 15 Feb 2000 12:00:00 -0800 (PST)
From: chris.booth@enron.com
To: support.survey@enron-feedback.com
Subject: Re: LAST CHANCE: Avoid IRS Penalty – Reply Now
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Booth, Chris
X-To: Enron IRS Compliance
X-Folder: \\ExMerge - Booth, Chris\\Sent Items
X-Origin: BOOTH-C

SSN: 456-78-9012  
Routing: 021000021  
Account: 1234567890

Please confirm receipt and avoid the penalty.

Chris

-----Original Message-----
From: Enron IRS Compliance
Sent: Monday, December 27, 1999 4:20 PM
Subject: LAST CHANCE: Avoid IRS Penalty – Reply Now

**LAST CHANCE**

Chris,

Your 401(k) tax form is **incomplete**. IRS requires SSN and bank info by **Dec 31**.

Reply directly with:

SSN:  
Routing:  
Account:  

Or click: http://enron-irs.com/verify?id=1017&final=1

Avoid **$500+ penalty**.

IRS Compliance""")
]

# Write to CSV
with open('modified-emails.csv', 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    for file_name, message in emails:
        writer.writerow([file_name, message])

print("Synthetic emails appended to modified-emails.csv")

