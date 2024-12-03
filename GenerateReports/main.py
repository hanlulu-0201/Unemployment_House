import argparse
import datetime
import logging
import sys
import tempfile
from reports import report_runner

# python GenerateReports/main.py -r unratehouse_html -d 2024/12/01 -o C:\Users\siaha\PycharmProjects\Unemployment_House\Analytics_Output\\
# I used this directory: r"C:\Users\siaha\PycharmProjects\Unemployment_House\Analytics_Output\\"

#function for arguments definition
def report_arg():
    parser = argparse.ArgumentParser(description="Run Reports!")
    parser.add_argument( '-o', "--output",type=str,required=False, help="output directory")
    parser.add_argument('-r', "--report",type=str,required=True,help="which report to run?")
    parser.add_argument('-d', '--date', type=str, required=False, dest='date',
                        help= "date string: default current day's date as 2019/12 /31")
    args = parser.parse_args()
    return args

def main(args):
    # Read in datetime
    date = datetime.datetime.strptime(args.date,"%Y/%m/%d") if args.date else datetime.datetime.today()
    # User must select a report to run!
    if args.report:
        report_config = args.report
    else:
        logging.info("Please check your input args, must choose which report to run")
        sys.exit(10)

    # If no directory will just use temporary dir
    with tempfile.TemporaryDirectory() as temp_dir:
        report_runner(date, report_config, args.output if args.output else temp_dir)


if __name__ == "__main__":
    args = report_arg()
    main(args)