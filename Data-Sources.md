# Data Sources

## Current Project Data

- `data/cms_fee_schedule.csv`
  - A project-formatted CMS-style Medicare fee schedule benchmark used by the app for CPT/HCPCS lookups.
  - It includes 9,926 rows with code, description, and fee columns.
  - It combines CMS Physician Fee Schedule, Clinical Laboratory Fee Schedule, and anesthesia reference data.
  - It is not a complete clinical billing database or a guarantee of patient-specific fair pricing.

- `data/sample_bill.txt`
  - A synthetic Duke-style bill used for demo and testing.
  - It is not a real patient bill.

- `data/bill images/`
  - Redacted or altered medical bill images and PDFs used for local OCR testing, qualitative testing, and demo inspiration.
  - These files were gathered from public posters with permission, and identifying names were changed or removed before inclusion.

- `data/supplemental info/`
  - Supplemental public reference materials about medical bills, itemized bills, explanations of benefits, and denial notices.
  - These files support qualitative understanding of bill formats and patient-facing explanations.

- `data/importance.png`
  - Supporting visual/reference material used to motivate the medical billing problem.

- `eval/evaluate_rules.py`
  - Uses synthetic evaluation cases to test deterministic overcharge, duplicate, and upcoding checks.
  - These cases are useful for reproducible evaluation, but they should not be described as real clinical billing data.

## Official Sources To Cite Or Expand Toward

- CMS Physician Fee Schedule documentation and files:
  - https://www.cms.gov/apps/physician-fee-schedule/documentation.aspx
  - CMS states that the Medicare Physician Fee Schedule includes more than 10,000 physician services, RVUs, payment policy indicators, and fee schedule amounts.

- CMS Physician Fee Schedule dataset explorer:
  - https://pfs.data.cms.gov/datasets
  - Useful for current and historical Medicare Physician Fee Schedule datasets, including indicators and localities.

- CMS HCPCS Quarterly Update:
  - https://www.cms.gov/medicare/coding-billing/healthcare-common-procedure-system/quarterly-update
  - CMS publishes official public use files for HCPCS Level II quarterly updates.

## Important Licensing Note

CPT code descriptions are maintained by the American Medical Association and may have licensing restrictions. In the project writeup, describe the current file as a CMS-style educational benchmark and avoid claiming that the repository redistributes a complete official CPT database.
