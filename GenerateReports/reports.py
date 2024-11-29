import sys
from Reporting import UnrateHouse

def unratehouse_pdf(date :str, tempdir: str):
    return UnrateHouse.generate_pdf_report(date, tempdir)

def unratehouse_excel(date :str, tempdir: str):
    return UnrateHouse.generate_excel_report(date, tempdir)


def report_runner(date :str, report_config: str, tempdir: str):
    return getattr(sys.modules[__name__], report_config)(date, tempdir)