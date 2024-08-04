import csv
import json
import os
from dataclasses import dataclass

from constants import SOURCES_ROOT
from utils import get_experiment_file, FileType


@dataclass
class Contact:
    # Transmitting node, from node
    tx_node: str
    # Receiving node, to node
    rx_node: str
    start_time: int
    end_time: int
    # The context attribute can be any additional information associated with the contact. For example this could
    # include the range or distance between the contacts, one way light time (OWLT), data rate, BER, etc
    context: dict


@dataclass
class ContactPlan:
    contacts: list[Contact]


class IONContactPlanParser:
    
    CONTACT_PREFIX = ["a", "contact"]
    TIMESTAMP_PREFIX = "+"
    DURATION_CONTEXT = "duration"
    RANGE_CONTEXT = "range"

    def read(self, experiment_name: str) -> ContactPlan:
        contacts = []

        path = get_experiment_file(experiment_name, FileType.CONTACT_PLAN)
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                if not row:
                    continue
                
                ion_start_time, ion_end_time, tx_node, rx_node, duration, contact_range = row[2:]

                contact = Contact(
                    tx_node=tx_node,
                    rx_node=rx_node,
                    start_time=int(ion_start_time[1:]),
                    end_time=int(ion_end_time[1:]),
                    context={
                        IONContactPlanParser.DURATION_CONTEXT: float(duration),
                        IONContactPlanParser.RANGE_CONTEXT: float(contact_range)
                    }
                )

                contacts.append(contact)

        return ContactPlan(contacts)

    def write(self, experiment_name: str, contact_plan: ContactPlan, file_type: FileType):
        rows = []

        for contact in contact_plan.contacts:
            ion_start_time = f"{IONContactPlanParser.TIMESTAMP_PREFIX}{contact.start_time}"
            ion_end_time = f"{IONContactPlanParser.TIMESTAMP_PREFIX}{contact.end_time}"

            row = [
                ion_start_time,
                ion_end_time,
                contact.tx_node,
                contact.rx_node,
            ]
            
            # Add optional context values to the row
            if IONContactPlanParser.DURATION_CONTEXT in contact.context:
                row.append(contact.context[IONContactPlanParser.DURATION_CONTEXT])
            if IONContactPlanParser.RANGE_CONTEXT in contact.context:
                row.append(contact.context[IONContactPlanParser.RANGE_CONTEXT])
            
            rows.append(IONContactPlanParser.CONTACT_PREFIX + row)

        path = get_experiment_file(experiment_name, file_type)
        with open(path, "w") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerows(rows)


class IPNDContactPlanParser:

    def write(self, experiment_name: str, contact_plan: ContactPlan):
        contact_plan_json = {
            "ContactPlan": []
        }
        
        initial_start_time = 725803264.184
        for contact in contact_plan.contacts:
            contact_json = {
                "SourceID": int(contact.tx_node),
                "DestinationID": int(contact.rx_node),
                "StartTime": initial_start_time + contact.start_time,
                "EndTime": initial_start_time + contact.end_time,
                "Duration": float(contact.end_time - contact.start_time),
                "Color": []
            }
            contact_plan_json["ContactPlan"].append(contact_json)

        path = os.path.join(SOURCES_ROOT, experiment_name, "contactPlan.json")
        with open(path, "w") as f:
            json.dump(contact_plan_json, f, indent=4)
