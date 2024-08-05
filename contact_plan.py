import csv
import json
import os
from dataclasses import dataclass

from constants import SOURCES_ROOT
from utils import get_experiment_file, FileType


@dataclass
class Contact:
    tx_node: str  # Transmitting node, from node
    rx_node: str  # Receiving node, to node
    start_time: int
    end_time: int
    bit_rate: int  # bits per second
    range: float  # distance between the nodes in light-seconds


@dataclass
class ContactPlan:
    contacts: list[Contact]


class IONContactPlanParser:
    
    CONTACT_PREFIX = ["a", "contact"]
    RANGE_PREFIX = ["a", "range"]

    TIMESTAMP_PREFIX = "+"

    def read(self, experiment_name: str) -> ContactPlan:
        contacts = []

        path = get_experiment_file(experiment_name, FileType.CONTACT_PLAN)
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            
            # We assume here that the contact plan has each contact written as "a contact" command followed by "a range"
            # command as the next row
            iter_reader = iter(reader)
            for contact_row, range_row in zip(iter_reader, iter_reader):
                ion_start_time, ion_end_time, tx_node, rx_node, bit_rate = contact_row[2:]
                range_in_light_seconds = range_row[-1]

                contact = Contact(
                    tx_node=tx_node,
                    rx_node=rx_node,
                    start_time=int(ion_start_time[1:]),
                    end_time=int(ion_end_time[1:]),
                    bit_rate=int(bit_rate),
                    range=float(range_in_light_seconds),
                )

                contacts.append(contact)

        return ContactPlan(contacts)

    def write(self, experiment_name: str, contact_plan: ContactPlan, file_type: FileType):
        contact_rows = []
        range_rows = []

        for contact in contact_plan.contacts:
            ion_start_time = f"{IONContactPlanParser.TIMESTAMP_PREFIX}{contact.start_time}"
            ion_end_time = f"{IONContactPlanParser.TIMESTAMP_PREFIX}{contact.end_time}"

            contact_row = [
                ion_start_time,
                ion_end_time,
                contact.tx_node,
                contact.rx_node,
                contact.bit_rate,
            ]
            contact_rows.append(IONContactPlanParser.CONTACT_PREFIX + contact_row)

            range_row = [
                ion_start_time,
                ion_end_time,
                contact.tx_node,
                contact.rx_node,
                contact.range,
            ]
            range_rows.append(IONContactPlanParser.RANGE_PREFIX + range_row)

        path = get_experiment_file(experiment_name, file_type)
        with open(path, "w") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerows(contact_rows)
            writer.writerows(range_rows)


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
