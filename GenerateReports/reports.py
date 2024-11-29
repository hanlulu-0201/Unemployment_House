import sys
from Reporting import UnrateHouseExcel
from Reporting import UnrateHousePDF

def UnrateHousePDFreport(date :str, tempdir: str):
    return UnrateHousePDF.generateReport(date, tempdir)

def UnrateHouseExcel_report(date :str, tempdir: str):
    return UnrateHouseExcel.generateReport(date, tempdir)


def report_runner(date :str, report_config: str, tempdir: str):
    return getattr(sys.modules[__name__], report_config)(date, tempdir)