import argparse
from contact_plan.contact_plan_parser import IONContactPlanParser


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', help='Contact plan file input name ')
    return parser.parse_args()


def main(args):
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(args.file_name)
    print(contact_plan)
    contact_plan_parser.write(args.file_name, contact_plan)


if __name__ == "__main__":
    args = get_args()
    main(args)
