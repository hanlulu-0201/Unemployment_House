import sys
from Reporting import UnrateHouse

def unratehouse_pdf(date :str, dir: str):
    return UnrateHouse.generate_pdf_report(date, dir)

def unratehouse_excel(date :str, dir: str):
    return UnrateHouse.generate_excel_report(date, dir)

def unratehouse_html(date :str, dir: str):
    return UnrateHouse.generate_html_report(date, dir)


def report_runner(date :str, report_config: str, dir: str):
    return getattr(sys.modules[__name__], report_config)(date, dir)