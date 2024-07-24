import argparse
from contact_plan.contact_plan_parser import IONContactPlanParser
from time_expanded_graph.time_expanded_graph import build_time_expanded_graph, write_time_expanded_graph


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', help='Contact plan file input name ')
    return parser.parse_args()


def main(file_name):
    # Read contact plan from disk
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(file_name)
    
    # Split long contacts into multiple smaller contacts, a maximum duration of 100s
    # Write split contact plan back to disk
    # TODO

    # Convert split contact plan into a time expanded graph (TEG)
    time_expanded_graph = build_time_expanded_graph(contact_plan)
    write_time_expanded_graph(time_expanded_graph, file_name)
    
    # Iterate through each of the k graphs and compute the maximal matching
    # As we step through the k graphs we want to optimize for our metric
    # For each interplanetary node minimize the absolute difference between the duration of contacts where the
    # interplanetary node is the rx and the duration of contacts where the interplanetary node is the tx
    # This will produce the TEG with the maximum Earth-bound network capacity
    # TODO
    
    # Convert the TEG back to a contact plan
    # TODO
    
    # Write scheduled contact plan to disk
    contact_plan_parser.write(file_name, contact_plan)


if __name__ == "__main__":
    args = get_args()
    main(args.file_name)
